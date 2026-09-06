// Razorpay Checkout for the support page.
//
// Uses only the publishable key id, which is designed to be exposed in the
// browser (the same value is named NEXT_PUBLIC_RAZORPAY_KEY_ID in the sibling
// OpenCareerAI project). The key secret is never referenced here and must
// never reach the client.
//
// Payments are created directly from Checkout rather than from a server-created
// order, because a static site has no backend to call orders.create() on.

(() => {
  const CHECKOUT_SDK = "https://checkout.razorpay.com/v1/checkout.js";

  let sdkPromise = null;

  function loadCheckout() {
    if (window.Razorpay) return Promise.resolve(window.Razorpay);
    if (sdkPromise) return sdkPromise;

    sdkPromise = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = CHECKOUT_SDK;
      s.async = true;
      s.onload = () => resolve(window.Razorpay);
      s.onerror = () => reject(new Error("Could not load Razorpay Checkout"));
      document.head.appendChild(s);
    });
    return sdkPromise;
  }

  function setStatus(root, message, kind) {
    const el = root.querySelector("[data-support-status]");
    if (!el) return;
    el.textContent = message || "";
    el.className = kind ? `support-status support-status--${kind}` : "support-status";
  }

  async function pay(root, rupees) {
    const keyId = root.dataset.razorpayKey;
    if (!keyId) {
      setStatus(root, "Payments are not configured yet.", "error");
      return;
    }

    const amount = Math.round(Number(rupees) * 100);
    if (!Number.isFinite(amount) || amount < 100) {
      setStatus(root, "Please enter an amount of ₹1 or more.", "error");
      return;
    }

    setStatus(root, "Opening secure checkout…", "pending");

    let Razorpay;
    try {
      Razorpay = await loadCheckout();
    } catch (err) {
      setStatus(
        root,
        "Checkout could not load. If you use a content blocker, allow " +
          "checkout.razorpay.com and try again.",
        "error"
      );
      return;
    }

    const rzp = new Razorpay({
      key: keyId,
      amount,
      currency: "INR",
      name: root.dataset.supportName || "AI/ML Mastery Program",
      description: "Voluntary contribution — free curriculum",
      image: root.dataset.supportLogo || undefined,
      notes: { purpose: "curriculum_support", source: "github_pages" },
      theme: { color: "#3f51b5" },
      handler(response) {
        setStatus(
          root,
          `Thank you. Payment ${response.razorpay_payment_id} received.`,
          "success"
        );
      },
      modal: {
        ondismiss() {
          setStatus(root, "Checkout closed — nothing was charged.", "");
        }
      }
    });

    rzp.on("payment.failed", (response) => {
      const reason = response?.error?.description || "The payment did not go through.";
      setStatus(root, `${reason} Nothing was charged.`, "error");
    });

    rzp.open();
  }

  function wire(root) {
    if (root.dataset.supportWired === "1") return;
    root.dataset.supportWired = "1";

    root.querySelectorAll("[data-support-amount]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        pay(root, btn.dataset.supportAmount);
      });
    });

    const form = root.querySelector("[data-support-custom]");
    if (form) {
      form.addEventListener("submit", (e) => {
        e.preventDefault();
        const input = form.querySelector("input[type=number]");
        pay(root, input && input.value);
      });
    }
  }

  // navigation.instant swaps page content without a reload, so re-wire on each
  // page render rather than once on DOMContentLoaded.
  const boot = () => document.querySelectorAll("[data-support-widget]").forEach(wire);

  if (typeof document$ !== "undefined") {
    document$.subscribe(boot);
  } else {
    document.addEventListener("DOMContentLoaded", boot);
  }
})();
