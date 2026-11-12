# Nghiên cứu ArtGS cùng AI Agent trên Google Colab

Hướng dẫn dùng repo này để làm nghiên cứu (tái lập ArtGS, thử ý tưởng cải tiến, đánh giá)
với một AI agent (Claude Code) điều phối từ máy local, GPU chạy trên **Google Colab** qua
[Colab CLI](https://pypi.org/project/google-colab-cli/), dữ liệu và cache lưu trên
**Google Drive** qua rclone.

> Agent đọc [`CLAUDE.md`](../CLAUDE.md) ở mỗi phiên: đó là "hợp đồng" (lệnh khởi động, phiên bản,
> đường dẫn, quy tắc). File này dành cho người.

## 1. Kiến trúc

```
 Máy local (Linux/macOS)                 Google Colab VM (T4/L4/A100)          Google Drive (NCKH-2027)
 ─────────────────────────               ──────────────────────────────        ────────────────────────
 Claude Code (agent)                      /content/venv  Python 3.10           data/archives/*.zip (+SHA256SUMS)
 colab CLI ── colab new/stop ──────────▶  torch 2.1.2+cu121 + 5 CUDA ext   ◀──  _env_cache/wheels/<key>/
 ssh/rsync ── code (rsync) ────────────▶  /content/ArtGS  (code)               data/processed/, data/raw/
 runs/<RUN_ID>/  ◀── log, metrics ──────  /content/data   (dataset, ổ local) ◀── checkpoints/ArtGS_ckpt/
 results/leaderboard.md                   /content/runs/<RUN_ID>  ────────────▶ experiments/<RUN_ID>/
```

- **Code** đi bằng `rsync` (kể cả thay đổi chưa commit), không cần token GitHub trên VM.
- **Môi trường** dựng bởi `scripts/bootstrap_colab.sh`; 5 CUDA extension build một lần rồi lưu
  wheel lên Drive, các lần sau chỉ cài wheel (không build lại trừ khi đổi torch/CUDA/GPU/source).
- **Dữ liệu** kéo từ Drive (hoặc Hugging Face nếu Drive chưa có) về ổ đĩa VM, kiểm sha256;
  train không bao giờ đọc trực tiếp từ Drive.
- **Mỗi thí nghiệm** là một "run" có ghi chép đầy đủ (commit, diff, môi trường, log, metric) và
  VM **luôn được tắt** khi xong, kể cả khi lỗi.

## 2. Cài đặt một lần

Yêu cầu: Linux hoặc macOS (Colab CLI không hỗ trợ Windows native; dùng WSL), `git`, `curl`,
tài khoản Google có Colab (có compute units nếu cần GPU), Claude Code.

```bash
git clone --recursive git@github.com:dinhxuanhuy/ArtGS.git && cd ArtGS
bash scripts/setup_local.sh                 # cài/cập nhật uv + Colab CLI, copy skill cho agent
colab --auth=oauth2 sessions                # đăng nhập Google cho Colab (mở URL, dán mã)
```

**Google Drive (rclone)**: `colab drivemount` cần bấm duyệt trên trình duyệt ở mỗi VM nên không
dùng được cho agent; repo dùng rclone. Trên máy có trình duyệt chạy `rclone authorize "drive"`,
rồi trên máy local:

```bash
rclone config create nckh drive scope=drive root_folder_id=<ID_folder_Drive> token='<token JSON>'
rclone lsf nckh:                            # kiểm tra
```

> Nên tạo **client_id riêng** (https://rclone.org/drive/#making-your-own-client-id): client_id
> dùng chung của rclone đang bị Google ngừng dần trong 2026. Không có rclone, mọi thứ vẫn chạy
> nhưng không có cache (mỗi VM build lại extension ~15-25 phút và tải data từ HF).

**Hook cho agent (khuyên dùng)**: `cp scripts/hooks/settings.proposed.json .claude/settings.json`
rồi mở `/hooks` trong Claude Code. Hook sẽ cảnh báo khi còn VM Colab đang chạy và chặn commit
data/checkpoint/token.

**Đưa dữ liệu lên Drive** (chỉ người có bản gốc làm, một lần):
`bash scripts/drive_publish_data.sh stage && bash scripts/drive_publish_data.sh upload && bash scripts/drive_publish_data.sh check`

## 3. Làm việc hằng ngày với agent

Mở Claude Code trong thư mục repo (`claude`). Agent tự đọc `CLAUDE.md`. Ví dụ yêu cầu:

| Bạn nói | Agent làm |
|---|---|
| "Chạy smoke test PARIS trên T4" | `colab_job.sh scripts/jobs/smoke_paris.sh` -> báo log, metric, thời gian, CU đã dùng |
| "Tạo baseline từ checkpoint chính thức" | `colab_job.sh scripts/jobs/eval_official.sh --datasets "paris dta artgs ckpt"` -> `results/leaderboard.md` |
| "Thử ý tưởng X trên foldchair và washer, 3 seed" | tạo nhánh `exp/x`, sửa code, viết job, chạy, so với baseline, ghi `research/journal.md` |
| "Kết quả run hôm qua thế nào?" | đọc `runs/<RUN_ID>/metrics.json`, bảng leaderboard (skill `analyze-results`) |
| "Còn VM nào chạy không?" | `colab sessions`, tắt nếu có |

Skill có sẵn trong `.claude/skills/`: `colab` (thao tác Colab CLI), `run-experiment`
(quy trình chạy thí nghiệm), `analyze-results` (so sánh với baseline, ghi nhật ký).

Mẹo: giao việc theo **giả thuyết + tiêu chí đánh giá + ngân sách**, ví dụ: *"Giả thuyết: thêm
loss X giảm lỗi trục khớp. Chạy blade và storage, 3 seed, tối đa 5 CU. So với baseline
official, báo mean ± std."*

## 4. Lệnh thủ công (khi không dùng agent)

```bash
colab usage                                                  # số dư compute units
bash scripts/colab_job.sh scripts/jobs/smoke_paris.sh --gpu T4 --datasets "paris" --max-cu 5
# VM tương tác tự quản lý:
bash scripts/colab_up.sh artgs-t4 --gpu T4                   # tạo VM + bootstrap + data
ssh -F scripts/ssh_config artgs-t4.colab                     # shell trên VM (hoặc: colab ssh -s artgs-t4)
rsync -az -e "ssh -F scripts/ssh_config" ./ artgs-t4.colab:/content/ArtGS/ --exclude-from=scripts/rsync_excludes.txt
bash scripts/colab_down.sh artgs-t4                          # LUÔN tắt khi xong (hoặc --all)
python3 scripts/leaderboard.py                               # cập nhật results/leaderboard.md
```

Viết job mới: copy `scripts/jobs/smoke_paris.sh`. Job chạy trên VM trong `/content/ArtGS` với
venv đã kích hoạt; ghi **mọi** output vào `$RUN_OUT`, theo layout
`$RUN_OUT/<dataset>/<subset>/<scene>/<model>` để `collect_metrics.py` tìm thấy `result.csv`.

## 5. Pipeline ArtGS và metric

1. `train_coarse.py` 10000 iter (bắt buộc đủ 10000: `train.py` nạp `coarse_gs/.../iteration_10000`)
2. `train_predict.py` dự đoán loại khớp
3. `train.py` 20000 iter, `--use_art_type_prior`
4. `render.py --iteration best --skip_test` -> `result.csv` (`best` chỉ có khi train > 15000 iter)

Metric (càng thấp càng tốt): `avg_angle` lỗi góc trục (độ), `avg_distance` lỗi vị trí trục,
`avg_theta_diff` lỗi chuyển động bộ phận, `avg_CD_static/dynamic/whole` Chamfer distance.
Baseline = checkpoint chính thức **chạy lại trong môi trường của mình**, không lấy số từ paper.

## 6. Dữ liệu, ghi chép, cache

| Gì | Ở đâu |
|---|---|
| Danh mục dataset (URL, kích thước, sha256) | `configs/datasets.json`; tải: `python3 scripts/load_data.py paris dta artgs` |
| Dataset trên VM | `/content/data/<dataset>/<subset>/<scene>` |
| Bản ghi mỗi run | `runs/<RUN_ID>/` (job.sh, git_sha, git_diff.patch, env.txt, job.log, metrics.json) |
| Output đầy đủ (checkpoint) | Drive `experiments/<RUN_ID>/` |
| Bảng kết quả | `results/leaderboard.md` (sinh tự động, không commit) |
| Nhật ký nghiên cứu | `research/journal.md`, `research/ideas.md` (riêng tư, không commit) |

**Quy ước cache** cho kết quả trung gian (mask SAM, nhãn VLM, Gaussian khởi tạo...):
`cache/<stage>/<object_id>/<config_hash>/` qua `tools/cache.py`. `config_hash` là sha256 của
**mọi** tham số ảnh hưởng tới stage (model + version, prompt, ngưỡng...). Khác hash thì không
bao giờ dùng lại.

```python
from tools.cache import CacheEntry
e = CacheEntry("sam_masks", "paris/sapien/washer_103776", {"model": "sam2.1_l", "iou": 0.88})
if not e.complete:
    e.prepare(); run(out=e.path); e.mark_complete()
```

## 7. Môi trường (đã ghim)

Python 3.10 · torch 2.1.2+cu121 · numpy 1.26.4 · pillow 10.4.0 · open3d 0.18 · CUDA 12.1 + g++-11
(chỉ để build) · diff-gaussian-rasterization (fork ArtGS) · simple-knn · pointnet2 ·
pytorch3d V0.7.8 · tiny-cuda-nn @3daa6e5. Colab có sẵn Python 3.13 / torch 2.14 nhưng **không
dùng**: Pillow >= 11 làm hỏng `dataset_readers.py`, open3d/pytorch3d/tcnn chưa kiểm chứng.

Thời gian bootstrap (đo trên container Ubuntu 24.04 mô phỏng Colab; số đo trên T4 sẽ cập nhật):
lần đầu build ~8 phút với 20 CPU (T4 Colab có 2 CPU nên lâu hơn đáng kể); khi đã có wheel cache:
**~2 phút** đến lúc `verify_env.py` báo ALL OK.

## 8. Quy tắc (agent và người đều tuân theo)

- **Luôn tắt VM Colab** khi job xong hoặc lỗi; trước khi kết thúc phiên chạy `colab sessions`.
- Ngân sách mặc định 5 CU/job; hỏi trước khi vượt, khi dùng A100/H100, khi tải > 5 GB.
- Không commit secret/token/credential, dataset, checkpoint, cache, runs, results.
- Repo là **public**: code nghiên cứu chưa công bố giữ ở local hoặc repo private.
- Không sửa code lõi của ArtGS khi làm hạ tầng; thay đổi nghiên cứu đặt trên nhánh `exp/<ý-tưởng>`.
- Một bước lỗi hai lần thì dừng và báo cáo, không lặp mù.

## 9. Xử lý sự cố

| Triệu chứng | Nguyên nhân / cách sửa |
|---|---|
| `colab drivemount` treo hoặc lỗi 400 | Cần bấm duyệt trên trình duyệt mỗi VM -> dùng rclone (mục 2) |
| Build lỗi `pybind11/cast.h: expected template-name` | nvcc 12.1 + g++-12 không tương thích -> bootstrap dùng g++-11 |
| `TypeError: Cannot handle this data type ... |i1` khi load ảnh | Pillow >= 11 -> phải là 10.4.0 |
| `libGL.so.1` / `libgthread` không tìm thấy | bootstrap tự `apt-get install libgl1 libglib2.0-0t64` |
| `colab new --gpu ...` lỗi 400 | Hết quota/không có quyền GPU đó -> thử `--gpu T4` |
| rclone báo client_id dùng chung sắp ngừng | Tạo client_id riêng, `rclone config reconnect nckh:` |
| CUDA OOM khi render res 1 trên GPU 8 GB | Dùng `--resolution 2` và `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
