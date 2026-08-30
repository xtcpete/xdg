# XDG project page

Static, dependency-free project page for **XDG: Accelerated Visual Disambiguation**.
It is designed to be published from the `gh-pages` branch of
[`xtcpete/xdg`](https://github.com/xtcpete/xdg).

## Preview locally

From this directory, run:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>.

The page includes the paper PDF, author links, original paper figures, cropped
result tables, BibTeX, and static qualitative reconstructions from
AerialMegaDepth, VisymScenes, and WRIVA.

## Publish on `gh-pages`

Copy the **contents** of this directory to the root of the repository's
`gh-pages` branch. GitHub Pages should publish from the branch root. The included
`.nojekyll` file keeps asset paths unchanged.
