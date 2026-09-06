#!/usr/bin/env bash
# ==============================================================================
# trim_to_gif.sh
# ------------------------------------------------------------------------------
# High-Quality Video-to-GIF Converter & Splitter with Terminal Progress & Size Cap
#
# Supported Operating Modes:
#   1. AUTO-SPLIT MODE (--split):
#      Slices the ENTIRE video from 00:00:00 to the end into sequential GIF parts
#      (part_01.gif, part_02.gif, ...), guaranteeing EVERY part is strictly < 10 MB!
#
#   2. SINGLE TRIM MODE:
#      Trims a specific segment (-ss start, -t duration or -to end) into one GIF.
#
#   3. BATCH FOLDER MODE (--batch):
#      Processes all videos in a folder.
# ==============================================================================

set -o pipefail

# ANSI Color Codes
CLR_RESET="\033[0m"
CLR_BOLD="\033[1m"
CLR_DIM="\033[2m"
CLR_RED="\033[1;31m"
CLR_GREEN="\033[1;32m"
CLR_YELLOW="\033[1;33m"
CLR_BLUE="\033[1;34m"
CLR_MAGENTA="\033[1;35m"
CLR_CYAN="\033[1;36m"
CLR_WHITE="\033[1;37m"

print_banner() {
    echo -e "${CLR_CYAN}${CLR_BOLD}"
    echo "======================================================================"
    echo "   🎬  VIDEO TO GIF TRIMMER & AUTO-SPLITTER (<10 MB PER PART)         "
    echo "======================================================================"
    echo -e "${CLR_RESET}"
}

print_usage() {
    print_banner
    echo -e "${CLR_BOLD}USAGE MODES:${CLR_RESET}"
    echo ""
    echo -e "  ${CLR_GREEN}1) AUTO-SPLIT WHOLE VIDEO INTO PARTS (<10 MB each):${CLR_RESET}"
    echo "     ./trim_to_gif.sh --split <video.mp4> [segment_seconds] [output_folder] [max_mb]"
    echo "     e.g: ./trim_to_gif.sh --split video.mp4 4"
    echo "     (Splits video from start to end into 4-sec parts, all <10 MB)"
    echo ""
    echo -e "  ${CLR_CYAN}2) INTERACTIVE MODE (Prompted step-by-step):${CLR_RESET}"
    echo "     ./trim_to_gif.sh"
    echo "     (Asks whether you want to split whole video into parts or trim one part)"
    echo ""
    echo -e "  ${CLR_YELLOW}3) SINGLE TRIM MODE:${CLR_RESET}"
    echo "     ./trim_to_gif.sh -i <video.mp4> -ss <start> -t <duration> -o <output.gif> [-m <max_mb>]"
    echo "     e.g: ./trim_to_gif.sh -i video.mp4 -ss 00:00:02 -t 5 -o clip.gif"
    echo ""
    echo -e "  ${CLR_MAGENTA}4) BATCH FOLDER MODE:${CLR_RESET}"
    echo "     ./trim_to_gif.sh --batch <folder_path> [-m <max_mb>]"
    echo ""
    echo -e "${CLR_BOLD}OPTIONS:${CLR_RESET}"
    echo "  --split          Auto-split whole video into sequential parts"
    echo "  --chunk-sec      Duration of each part in seconds (Default: 4 seconds)"
    echo "  -i,  --input     Input video file path (.mp4, .mov, .mkv, .webm, etc.)"
    echo "  -ss, --start     Start time (e.g. 00:00:05, 5) [Default: 00:00:00]"
    echo "  -t,  --duration  Duration in seconds or HH:MM:SS"
    echo "  -to, --end       End time in seconds or HH:MM:SS"
    echo "  -o,  --output    Output file or folder path"
    echo "  -m,  --max-mb    Maximum target file size per GIF in MB [Default: 9.8 MB]"
    echo "  -w,  --width     Initial target width in px [Default: 480]"
    echo "  -fps             Target frame rate [Default: 12]"
    echo "  -h,  --help      Show this help documentation"
    exit 0
}

check_dependencies() {
    local missing=0
    if ! command -v ffmpeg &>/dev/null; then
        echo -e "${CLR_RED}❌ Error: 'ffmpeg' is not installed or not in PATH.${CLR_RESET}"
        echo "   Install via: brew install ffmpeg"
        missing=1
    fi
    if ! command -v ffprobe &>/dev/null; then
        echo -e "${CLR_RED}❌ Error: 'ffprobe' is not installed or not in PATH.${CLR_RESET}"
        echo "   Install via: brew install ffmpeg"
        missing=1
    fi
    if [ "$missing" -eq 1 ]; then
        exit 1
    fi
}

time_to_seconds() {
    local t="$1"
    t=$(echo "$t" | tr -d '[:space:]')
    if [ -z "$t" ]; then
        echo "0"
        return
    fi

    if [[ "$t" =~ ^([0-9]+):([0-9]{2}):([0-9]{2}(\.[0-9]+)?)$ ]]; then
        awk -v h="${BASH_REMATCH[1]}" -v m="${BASH_REMATCH[2]}" -v s="${BASH_REMATCH[3]}" 'BEGIN { printf "%.2f", h*3600 + m*60 + s }'
    elif [[ "$t" =~ ^([0-9]+):([0-9]{2}(\.[0-9]+)?)$ ]]; then
        awk -v m="${BASH_REMATCH[1]}" -v s="${BASH_REMATCH[2]}" 'BEGIN { printf "%.2f", m*60 + s }'
    elif [[ "$t" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        awk -v s="$t" 'BEGIN { printf "%.2f", s }'
    else
        echo "0"
    fi
}

format_seconds() {
    local total="$1"
    awk -v t="$total" 'BEGIN {
        h = int(t / 3600);
        m = int((t % 3600) / 60);
        s = t - (h*3600 + m*60);
        printf "%02d:%02d:%05.2f", h, m, s
    }'
}

get_file_size_bytes() {
    local f="$1"
    if [ ! -f "$f" ]; then
        echo 0
        return
    fi
    local sz
    sz=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || wc -c < "$f" | tr -d ' ')
    echo "${sz:-0}"
}

format_bytes() {
    local bytes="$1"
    awk -v b="$bytes" 'BEGIN {
        if (b >= 1048576) {
            printf "%.2f MB", b / 1048576;
        } else if (b >= 1024) {
            printf "%.1f KB", b / 1024;
        } else {
            printf "%d B", b;
        }
    }'
}

render_gif_with_progress() {
    local in_file="$1"
    local start_sec="$2"
    local dur_sec="$3"
    local out_file="$4"
    local width="$5"
    local fps="$6"
    local colors="$7"
    local label="$8"

    [ -z "$colors" ] && colors=128
    [ -z "$label" ] && label="Rendering GIF"

    local bar_width=26
    local filter="[0:v] fps=${fps},scale=${width}:-1:flags=lanczos,split [a][b];[a] palettegen=max_colors=${colors}:stats_mode=diff [p];[b][p] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"

    ffmpeg -y \
        -ss "$start_sec" \
        -t "$dur_sec" \
        -i "$in_file" \
        -filter_complex "$filter" \
        -progress pipe:1 \
        "$out_file" 2>/dev/null | while IFS='=' read -r key val; do
        if [ "$key" = "out_time_us" ]; then
            local curr_sec
            curr_sec=$(awk -v us="$val" 'BEGIN { printf "%.2f", us / 1000000 }')
            local pct
            pct=$(awk -v c="$curr_sec" -v t="$dur_sec" 'BEGIN { p = int((c/t)*100); if (p>100) p=100; if (p<0) p=0; print p }')
            local filled=$(( pct * bar_width / 100 ))
            local empty=$(( bar_width - filled ))
            local bar
            bar=$(printf "%0.s█" $(seq 1 $filled 2>/dev/null))
            local spacer
            spacer=$(printf "%0.s░" $(seq 1 $empty 2>/dev/null))
            printf "\r\033[K  ${CLR_CYAN}%s:${CLR_RESET} [%s%s] ${CLR_BOLD}%3d%%${CLR_RESET} (%.1fs / %.1fs)" \
                "$label" "$bar" "$spacer" "$pct" "$curr_sec" "$dur_sec"
        elif [ "$key" = "progress" ] && [ "$val" = "end" ]; then
            local full_bar
            full_bar=$(printf "%0.s█" $(seq 1 $bar_width 2>/dev/null))
            printf "\r\033[K  ${CLR_GREEN}%s:${CLR_RESET} [%s] ${CLR_BOLD}100%%${CLR_RESET} (%.1fs / %.1fs)\n" \
                "$label" "$full_bar" "$dur_sec" "$dur_sec"
        fi
    done

    [ -f "$out_file" ] && [ "$(get_file_size_bytes "$out_file")" -gt 0 ]
}

convert_segment_to_gif() {
    local in_file="$1"
    local start_sec="$2"
    local dur_sec="$3"
    local out_file="$4"
    local max_mb="$5"
    local width="$6"
    local fps="$7"
    local step_title="$8"

    [ -z "$max_mb" ] && max_mb="9.8"
    [ -z "$width" ] && width="480"
    [ -z "$fps" ] && fps="12"
    [ -z "$step_title" ] && step_title="Part"

    local max_bytes
    max_bytes=$(awk -v mb="$max_mb" 'BEGIN { printf "%d", mb * 1048576 }')

    local temp_gif="/tmp/trim_temp_$$.gif"
    local temp_opt="/tmp/trim_opt_$$.gif"
    rm -f "$temp_gif" "$temp_opt"

    local start_clock
    start_clock=$(date +%s)

    render_gif_with_progress "$in_file" "$start_sec" "$dur_sec" "$temp_gif" "$width" "$fps" 128 "$step_title"

    local curr_size
    curr_size=$(get_file_size_bytes "$temp_gif")

    local has_gifsicle=0
    command -v gifsicle &>/dev/null && has_gifsicle=1

    if [ "$curr_size" -le "$max_bytes" ]; then
        if [ "$has_gifsicle" -eq 1 ]; then
            gifsicle -O3 "$temp_gif" -o "$temp_opt" 2>/dev/null
            local opt_size
            opt_size=$(get_file_size_bytes "$temp_opt")
            if [ "$opt_size" -gt 0 ] && [ "$opt_size" -le "$curr_size" ]; then
                mv "$temp_opt" "$out_file"
                curr_size="$opt_size"
            else
                mv "$temp_gif" "$out_file"
            fi
        else
            mv "$temp_gif" "$out_file"
        fi
    else
        echo -e "  ${CLR_YELLOW}⚡ Initial size ($(format_bytes "$curr_size")) exceeds target (${max_mb} MB). Compressing...${CLR_RESET}"
        local reduced=0

        if [ "$has_gifsicle" -eq 1 ]; then
            gifsicle -O3 --lossy=80 "$temp_gif" -o "$temp_opt" 2>/dev/null
            curr_size=$(get_file_size_bytes "$temp_opt")
            if [ "$curr_size" -le "$max_bytes" ] && [ "$curr_size" -gt 0 ]; then
                mv "$temp_opt" "$out_file"
                reduced=1
            fi
        fi

        if [ "$reduced" -eq 0 ] && [ "$has_gifsicle" -eq 1 ]; then
            gifsicle -O3 --lossy=140 "$temp_gif" -o "$temp_opt" 2>/dev/null
            curr_size=$(get_file_size_bytes "$temp_opt")
            if [ "$curr_size" -le "$max_bytes" ] && [ "$curr_size" -gt 0 ]; then
                mv "$temp_opt" "$out_file"
                reduced=1
            fi
        fi

        if [ "$reduced" -eq 0 ]; then
            render_gif_with_progress "$in_file" "$start_sec" "$dur_sec" "$temp_gif" 380 10 96 "$step_title (Downscale)"
            curr_size=$(get_file_size_bytes "$temp_gif")
            if [ "$has_gifsicle" -eq 1 ]; then
                gifsicle -O3 --lossy=100 "$temp_gif" -o "$temp_opt" 2>/dev/null
                curr_size=$(get_file_size_bytes "$temp_opt")
                if [ "$curr_size" -le "$max_bytes" ] && [ "$curr_size" -gt 0 ]; then
                    mv "$temp_opt" "$out_file"
                    reduced=1
                fi
            elif [ "$curr_size" -le "$max_bytes" ]; then
                mv "$temp_gif" "$out_file"
                reduced=1
            fi
        fi

        if [ "$reduced" -eq 0 ]; then
            render_gif_with_progress "$in_file" "$start_sec" "$dur_sec" "$temp_gif" 320 8 64 "$step_title (Compact)"
            curr_size=$(get_file_size_bytes "$temp_gif")
            if [ "$has_gifsicle" -eq 1 ]; then
                gifsicle -O3 --lossy=160 "$temp_gif" -o "$temp_opt" 2>/dev/null
                curr_size=$(get_file_size_bytes "$temp_opt")
                mv "$temp_opt" "$out_file"
            else
                mv "$temp_gif" "$out_file"
            fi
            reduced=1
        fi
    fi

    if [ ! -f "$out_file" ] && [ -f "$temp_gif" ]; then
        mv "$temp_gif" "$out_file"
    fi

    rm -f "$temp_gif" "$temp_opt"
    local final_bytes
    final_bytes=$(get_file_size_bytes "$out_file")
    local end_clock
    end_clock=$(date +%s)
    local diff=$(( end_clock - start_clock ))

    echo -e "  ✅ ${CLR_BOLD}Saved:${CLR_RESET} $(basename "$out_file") | Size: ${CLR_GREEN}${CLR_BOLD}$(format_bytes "$final_bytes")${CLR_RESET} (target < ${max_mb} MB) [${diff}s]"
}

split_video_into_parts() {
    local in_file="$1"
    local chunk_sec="$2"
    local out_dir="$3"
    local max_mb="$4"
    local width="$5"
    local fps="$6"

    [ -z "$chunk_sec" ] && chunk_sec=4.0
    [ -z "$max_mb" ] && max_mb="9.8"
    [ -z "$width" ] && width="480"
    [ -z "$fps" ] && fps="12"

    local total_raw
    total_raw=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$in_file" 2>/dev/null)
    local total_sec
    total_sec=$(awk -v t="${total_raw:-0}" 'BEGIN { printf "%.2f", t }')

    local base_name
    base_name=$(basename "$in_file")
    local stem="${base_name%.*}"

    if [ -z "$out_dir" ]; then
        local in_dir
        in_dir=$(dirname "$in_file")
        out_dir="${in_dir}/${stem}_gif_parts"
    fi
    mkdir -p "$out_dir"

    local total_parts
    total_parts=$(awk -v tot="$total_sec" -v c="$chunk_sec" 'BEGIN {
        parts = int(tot / c);
        if (tot % c > 0.5) parts += 1;
        if (parts == 0) parts = 1;
        print parts;
    }')

    echo -e "🎬 ${CLR_BOLD}Input Video:${CLR_RESET}      ${CLR_WHITE}${base_name}${CLR_RESET}"
    echo -e "⏱️  ${CLR_BOLD}Total Length:${CLR_RESET}     $(format_seconds "$total_sec") (${total_sec}s)"
    echo -e "✂️  ${CLR_BOLD}Part Length:${CLR_RESET}      ${chunk_sec} seconds each"
    echo -e "🔢 ${CLR_BOLD}Total Parts:${CLR_RESET}      ${CLR_CYAN}${total_parts} GIF parts${CLR_RESET}"
    echo -e "🎯 ${CLR_BOLD}Size Limit:${CLR_RESET}       < ${max_mb} MB per part"
    echo -e "📁 ${CLR_BOLD}Output Directory:${CLR_RESET} ${out_dir}/"
    echo ""

    local overall_start
    overall_start=$(date +%s)
    local part_num=1
    local curr_start="0.00"

    while (( $(echo "$curr_start < $total_sec" | bc -l 2>/dev/null || echo 0) )); do
        local remaining
        remaining=$(awk -v tot="$total_sec" -v s="$curr_start" 'BEGIN { printf "%.2f", tot - s }')
        
        if (( $(echo "$remaining < 0.6" | bc -l 2>/dev/null || echo 0) )) && [ "$part_num" -gt 1 ]; then
            break
        fi

        local this_dur="$chunk_sec"
        if (( $(echo "$remaining < $chunk_sec" | bc -l 2>/dev/null || echo 0) )); then
            this_dur="$remaining"
        fi

        local part_padded
        part_padded=$(printf "%02d" "$part_num")
        local out_gif="${out_dir}/${stem}_part_${part_padded}.gif"

        echo -e "${CLR_MAGENTA}----------------------------------------------------------------------${CLR_RESET}"
        echo -e "🎬 ${CLR_BOLD}Part [${part_num}/${total_parts}]:${CLR_RESET} $(format_seconds "$curr_start") ➔ $(format_seconds "$(awk -v s="$curr_start" -v d="$this_dur" 'BEGIN { printf "%.2f", s+d }')") (${this_dur}s)"

        convert_segment_to_gif "$in_file" "$curr_start" "$this_dur" "$out_gif" "$max_mb" "$width" "$fps" "Part ${part_num}/${total_parts}"

        curr_start=$(awk -v s="$curr_start" -v d="$this_dur" 'BEGIN { printf "%.2f", s + d }')
        part_num=$(( part_num + 1 ))
    done

    local overall_end
    overall_end=$(date +%s)
    local overall_time=$(( overall_end - overall_start ))

    echo ""
    echo -e "${CLR_GREEN}${CLR_BOLD}======================================================================${CLR_RESET}"
    echo -e "${CLR_GREEN}${CLR_BOLD}  🎉  AUTO-SPLIT COMPLETE: ALL PARTS ARE < ${max_mb} MB!              ${CLR_RESET}"
    echo -e "${CLR_GREEN}${CLR_BOLD}======================================================================${CLR_RESET}"
    echo -e "  📁 Saved inside:    ${CLR_WHITE}${out_dir}/${CLR_RESET}"
    echo -e "  📦 Total Parts:     ${CLR_GREEN}${CLR_BOLD}$(( part_num - 1 )) GIF files${CLR_RESET}"
    echo -e "  ⏳ Total Time:      ${overall_time} seconds"
    echo -e "${CLR_GREEN}${CLR_BOLD}======================================================================${CLR_RESET}"
    echo ""
}

# ------------------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------------------
check_dependencies

MODE=""
INPUT_FILE=""
START_TIME="0"
DURATION=""
END_TIME=""
OUTPUT_FILE=""
MAX_MB="9.8"
WIDTH="480"
FPS="12"
CHUNK_SEC="4"
BATCH_DIR=""

if [ $# -gt 0 ]; then
    case "$1" in
        -h|--help)
            print_usage
            ;;
        --split)
            MODE="split"
            INPUT_FILE="$2"
            shift 2
            [ -n "$1" ] && CHUNK_SEC="$1" && shift
            [ -n "$1" ] && OUTPUT_FILE="$1" && shift
            [ -n "$1" ] && MAX_MB="$1" && shift
            ;;
        -b|--batch)
            MODE="batch"
            BATCH_DIR="$2"
            shift 2
            while [ $# -gt 0 ]; do
                case "$1" in
                    -m|--max-mb) MAX_MB="$2"; shift 2 ;;
                    *) shift ;;
                esac
            done
            ;;
        -i|--input|-ss|--start|-t|--duration|-to|--end|-o|--output|-m|--max-mb|-w|--width|-fps|--chunk-sec)
            while [ $# -gt 0 ]; do
                case "$1" in
                    -i|--input) INPUT_FILE="$2"; shift 2 ;;
                    --split) MODE="split"; shift ;;
                    --chunk-sec) CHUNK_SEC="$2"; shift 2 ;;
                    -ss|--start) START_TIME="$2"; shift 2 ;;
                    -t|--duration) DURATION="$2"; shift 2 ;;
                    -to|--end) END_TIME="$2"; shift 2 ;;
                    -o|--output) OUTPUT_FILE="$2"; shift 2 ;;
                    -m|--max-mb) MAX_MB="$2"; shift 2 ;;
                    -w|--width) WIDTH="$2"; shift 2 ;;
                    -fps) FPS="$2"; shift 2 ;;
                    *) echo -e "${CLR_RED}Unknown argument: $1${CLR_RESET}"; print_usage ;;
                esac
            done
            ;;
        *)
            INPUT_FILE="$1"
            [ -n "$2" ] && START_TIME="$2"
            [ -n "$3" ] && DURATION="$3"
            [ -n "$4" ] && OUTPUT_FILE="$4"
            [ -n "$5" ] && MAX_MB="$5"
            ;;
    esac
fi

print_banner

if [ "$MODE" = "split" ]; then
    if [ -z "$INPUT_FILE" ]; then
        echo -e "${CLR_BOLD}Enter path to video file to split into parts (drag & drop):${CLR_RESET}"
        read -r -p "  > " INPUT_FILE
        INPUT_FILE=$(echo "$INPUT_FILE" | sed -e "s/^['\"]//" -e "s/['\"]$//" | tr -d '\r')
    fi
    if [ ! -f "$INPUT_FILE" ]; then
        echo -e "${CLR_RED}❌ Error: File not found: '$INPUT_FILE'${CLR_RESET}"
        exit 1
    fi
    split_video_into_parts "$INPUT_FILE" "$CHUNK_SEC" "$OUTPUT_FILE" "$MAX_MB" "$WIDTH" "$FPS"
    exit 0
fi

if [ -z "$INPUT_FILE" ] && [ -z "$BATCH_DIR" ]; then
    echo -e "${CLR_BOLD}Enter path to video file (or drag & drop here):${CLR_RESET}"
    read -r -p "  > " INPUT_FILE
    INPUT_FILE=$(echo "$INPUT_FILE" | sed -e "s/^['\"]//" -e "s/['\"]$//" | tr -d '\r')

    if [ ! -f "$INPUT_FILE" ]; then
        echo -e "${CLR_RED}❌ Error: File not found: '$INPUT_FILE'${CLR_RESET}"
        exit 1
    fi

    TOTAL_DURATION_RAW=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$INPUT_FILE" 2>/dev/null)
    TOTAL_SEC=$(awk -v t="${TOTAL_DURATION_RAW:-0}" 'BEGIN { printf "%.2f", t }')

    echo -e "\n${CLR_BOLD}What would you like to do with this video? (${TOTAL_SEC}s total length):${CLR_RESET}"
    echo -e "  ${CLR_GREEN}1) Split ENTIRE video into sequential GIF parts (<10 MB each)${CLR_RESET} [Recommended]"
    echo -e "  ${CLR_CYAN}2) Trim only ONE specific section to a GIF (<10 MB)${CLR_RESET}"
    read -r -p "Select option [1 or 2] (Default 1): " user_choice
    [ -z "$user_choice" ] && user_choice="1"

    if [ "$user_choice" = "1" ]; then
        echo -e "\n${CLR_BOLD}Length for each GIF part in seconds? (Press Enter for 4 seconds):${CLR_RESET}"
        read -r -p "  > " user_chunk
        [ -n "$user_chunk" ] && CHUNK_SEC="$user_chunk"
        split_video_into_parts "$INPUT_FILE" "$CHUNK_SEC" "$OUTPUT_FILE" "$MAX_MB" "$WIDTH" "$FPS"
        exit 0
    fi
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${CLR_RED}❌ Error: File not found: '$INPUT_FILE'${CLR_RESET}"
    exit 1
fi

TOTAL_DURATION_RAW=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$INPUT_FILE" 2>/dev/null)
TOTAL_SEC=$(awk -v t="${TOTAL_DURATION_RAW:-0}" 'BEGIN { printf "%.2f", t }')
TOTAL_FMT=$(format_seconds "$TOTAL_SEC")

if [ -z "$START_TIME" ] || [ "$START_TIME" = "0" ]; then
    echo -e "${CLR_BOLD}Start time [HH:MM:SS or seconds] (Press Enter for 00:00:00):${CLR_RESET}"
    read -r -p "  > " user_start
    [ -n "$user_start" ] && START_TIME="$user_start"
fi

if [ -z "$DURATION" ] && [ -z "$END_TIME" ]; then
    echo -e "${CLR_BOLD}Duration or End Time [HH:MM:SS or seconds] (Press Enter for full remaining ${TOTAL_SEC}s):${CLR_RESET}"
    read -r -p "  > " user_dur
    [ -n "$user_dur" ] && DURATION="$user_dur"
fi

START_SEC=$(time_to_seconds "$START_TIME")
if [ -n "$DURATION" ]; then
    user_val_sec=$(time_to_seconds "$DURATION")
    if (( $(echo "$user_val_sec > $START_SEC" | bc -l 2>/dev/null || echo 0) )); then
        DUR_SEC=$(awk -v e="$user_val_sec" -v s="$START_SEC" 'BEGIN { printf "%.2f", e - s }')
    else
        DUR_SEC="$user_val_sec"
    fi
else
    DUR_SEC=$(awk -v tot="$TOTAL_SEC" -v s="$START_SEC" 'BEGIN { d = tot - s; if (d<=0) d=tot; printf "%.2f", d }')
fi

if [ -z "$OUTPUT_FILE" ]; then
    base_dir=$(dirname "$INPUT_FILE")
    base_name=$(basename "$INPUT_FILE")
    stem="${base_name%.*}"
    OUTPUT_FILE="${base_dir}/${stem}_trimmed.gif"
fi

echo -e "📁 ${CLR_BOLD}Input Video:${CLR_RESET}   ${CLR_WHITE}$(basename "$INPUT_FILE")${CLR_RESET}"
echo -e "⏱️  ${CLR_BOLD}Video Length:${CLR_RESET}  ${TOTAL_FMT} (${TOTAL_SEC}s)"
echo -e "✂️  ${CLR_BOLD}Trim Segment:${CLR_RESET}  $(format_seconds "$START_SEC") ➔ $(format_seconds "$(awk -v s="$START_SEC" -v d="$DUR_SEC" 'BEGIN { printf "%.2f", s+d }')") (${DUR_SEC}s duration)"
echo -e "🎯 ${CLR_BOLD}Target Size:${CLR_RESET}   < ${MAX_MB} MB"
echo -e "💾 ${CLR_BOLD}Output File:${CLR_RESET}   ${OUTPUT_FILE}\n"

convert_segment_to_gif "$INPUT_FILE" "$START_SEC" "$DUR_SEC" "$OUTPUT_FILE" "$MAX_MB" "$WIDTH" "$FPS" "Render GIF"
