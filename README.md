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

## Citation

```bibtex
@misc{chen2026xdgacceleratedvisualdisambiguation,
  title         = {XDG: Accelerated Visual Disambiguation},
  author        = {Gonglin Chen and Ben Southall and Hanyuan Xiao and Wenbin Teng and Haolin Xiong and Tianwen Fu and Junyi Ouyang and Kshitij Singh Minhas and Supun Samarasekera and Rakesh Kumar and Yajie Zhao},
  year          = {2026},
  eprint        = {2608.29733},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2608.29733}
}
```

## Publish on `gh-pages`

Copy the **contents** of this directory to the root of the repository's
`gh-pages` branch. GitHub Pages should publish from the branch root. The included
`.nojekyll` file keeps asset paths unchanged.
