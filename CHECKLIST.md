# PR-MaGIC Checklist

## Validation (GPU required)

### PerSAM-IL
- [ ] Run on FSS-1000 fold 0 and compare mIoU with original result
  ```bash
  cd /home/user11/PR-MaGIC
  python scripts/launch_persam_il.py --benchmarks fss --folds 0 --gpus 0
  ```
- [ ] Confirm results match `/home/user11/Personalize-SAM/save/` outputs
- [ ] Run remaining folds (1, 2, 3) and full benchmarks (coco, pascal, lvis)

### Matcher-IL
- [ ] Run on FSS-1000 fold 0 and compare mIoU with original result
  ```bash
  cd /home/user11/PR-MaGIC
  python scripts/launch_matcher_il.py --benchmarks fss --folds 0 --gpus 0
  ```
- [ ] Confirm results match `/home/user11/Matcher/save/` outputs
- [ ] Run remaining folds and full benchmarks

### DIS5K
- [ ] Verify `data/DIS5K` symlink works correctly for both PerSAM-IL and Matcher-IL
- [ ] Run DIS5K benchmark and confirm results

---

## Page (docs/index.html)
URL: https://postech-minjaelee.github.io/PR-MaGIC/

- [ ] Add teaser figure (`docs/static/images/teaser.jpg`)
- [ ] Add qualitative results figure (`docs/static/images/results.jpg`)
- [ ] Fill in Paper / arXiv links once published
- [ ] Add quantitative results table
- [ ] Polish layout and content

---

## README.md (after paper finalization)
- [ ] Write README with paper title, method overview, install instructions, run commands, results table
- [ ] Push to GitHub

---

## Notes
- Original files (reference only, do NOT modify):
  - `/home/user11/Personalize-SAM/`
  - `/home/user11/Matcher/`
- PR-MaGIC repo: `/home/user11/PR-MaGIC/`
- GitHub: https://github.com/POSTECH-MinjaeLee/PR-MaGIC
- Weights already downloaded: `PR-MaGIC/weights/sam_vit_h_4b8939.pth`, `dinov2_vitl14_pretrain.pth`
