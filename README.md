# NCKH-2027: Tái tạo vật thể có khớp (Articulated Object Reconstruction)

Repo nghiên cứu về **dựng lại vật thể có khớp** (tủ, ngăn kéo, laptop, kéo...) từ ảnh đa góc:
hình học từng bộ phận + tham số khớp (loại khớp, trục, trạng thái chuyển động) để tạo bản sao
số tương tác được. Phương pháp nền (baseline) là **ArtGS** (ICLR 2025, 3D Gaussian Splatting);
repo đóng gói một quy trình nghiên cứu **tái lập được**, chạy trên **GPU Google Colab** và có thể
giao cho **AI agent** (Claude Code) điều phối.

## Repo có gì

- **Baseline ArtGS** đầy đủ: `train_coarse.py` -> `train_predict.py` -> `train.py` -> `render.py`.
- **Môi trường ghim phiên bản** (Python 3.10, torch 2.1.2+cu121, 5 CUDA extension) dựng tự động
  trên VM Colab; extension build một lần rồi cache wheel trên Google Drive.
- **Dữ liệu có kiểm soát**: danh mục dataset (PARIS, DTA-Multi, ArtGS-Multi, checkpoint chính thức)
  với sha256, tải và giải nén bằng một lệnh.
- **Thí nghiệm một lệnh**: tạo VM -> dựng môi trường -> chạy job -> kéo log/metric về -> **tắt VM**,
  có giới hạn compute unit và ghi chép đầy đủ (commit, diff, môi trường, metric).
- **Đánh giá và so sánh**: `metrics.json` cho mỗi run, bảng leaderboard so với baseline chạy lại
  trong cùng môi trường.
- **Hướng dẫn cho AI agent**: [`CLAUDE.md`](CLAUDE.md) + skill trong `.claude/skills/`.

## Bắt đầu nhanh

```bash
git clone --recursive git@github.com:dinhxuanhuy/ArtGS.git && cd ArtGS
bash scripts/setup_local.sh                     # Colab CLI + kiểm tra đăng nhập
colab --auth=oauth2 sessions                    # đăng nhập Google (một lần)
# (khuyên dùng) cấu hình rclone remote "nckh" tới folder Drive dự án, xem docs/AGENT_RESEARCH.md
bash scripts/colab_job.sh scripts/jobs/smoke_paris.sh --gpu T4 --datasets "paris"
```

Làm việc với agent: mở `claude` trong thư mục repo và giao việc, ví dụ *"chạy baseline từ
checkpoint chính thức trên PARIS"* hoặc *"thử ý tưởng X trên 2 object, 3 seed, tối đa 5 CU"*.

## Tài liệu

| Tài liệu | Nội dung |
|---|---|
| [`docs/AGENT_RESEARCH.md`](docs/AGENT_RESEARCH.md) | Hướng dẫn đầy đủ: kiến trúc, cài đặt, quy trình hằng ngày với agent, dữ liệu, cache, xử lý sự cố |
| [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) | Paper liên quan: ArtGS, REArtGS, REArtGS++, Rest2Art |
| [`CLAUDE.md`](CLAUDE.md) | "Hợp đồng" cho AI agent: lệnh, phiên bản, đường dẫn, quy tắc |
| [`docs/ARTGS_README.md`](docs/ARTGS_README.md) | README gốc của ArtGS (cài đặt, dữ liệu, huấn luyện, công cụ trực quan hoá) |

## Cấu trúc

```
train_coarse.py, train_predict.py, train.py, render.py   pipeline ArtGS (baseline)
scene/ gaussian_renderer/ utils/ arguments/ submodules/  mã nguồn ArtGS + CUDA extension
scripts/
  setup_local.sh  bootstrap_colab.sh  colab_up.sh  colab_down.sh  colab_job.sh   Colab
  load_data.py  drive_publish_data.sh                                             dữ liệu
  collect_metrics.py  leaderboard.py                                              kết quả
  jobs/            mẫu job (smoke test, baseline từ checkpoint chính thức)
  hooks/           hook cho Claude Code (cảnh báo VM còn chạy, chặn commit dữ liệu/secret)
tools/cache.py     cache kết quả trung gian: cache/<stage>/<object_id>/<config_hash>/
configs/datasets.json   danh mục dataset (URL, kích thước, sha256)
docs/              hướng dẫn và tài liệu tham khảo
```

Không commit: dữ liệu, checkpoint, cache, `runs/`, `results/`, token. Ghi chép nghiên cứu
(`research/`) để ở local.

## Ghi công và giấy phép

Mã nguồn phương pháp nền là **ArtGS** của Yu Liu, Baoxiong Jia, Ruijie Lu, Junfeng Ni,
Song-Chun Zhu, Siyuan Huang ([YuLiu-LY/ArtGS](https://github.com/YuLiu-LY/ArtGS),
[paper](https://arxiv.org/abs/2502.19459)), phát hành theo giấy phép Apache-2.0 ([`LICENSE`](LICENSE)).
Repo này giữ nguyên giấy phép đó cho phần mã gốc. Nếu dùng ArtGS, hãy trích dẫn:

```bibtex
@inproceedings{liu2025building,
  title={Building Interactable Replicas of Complex Articulated Objects via Gaussian Splatting},
  author={Liu, Yu and Jia, Baoxiong and Lu, Ruijie and Ni, Junfeng and Zhu, Song-Chun and Huang, Siyuan},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
}
```
