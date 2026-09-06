window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// `navigation.instant` swaps page content via XHR without a document reload,
// so MathJax must be re-run on every navigation or equations stay as raw
// \(...\) source. document$ is the observable Material emits per page load.
document$.subscribe(() => {
  if (typeof MathJax === "undefined" || !MathJax.startup) {
    return;
  }
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
