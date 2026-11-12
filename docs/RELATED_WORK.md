# Related work (tài liệu tham khảo)

PDF + bản text trích xuất của các paper dưới đây nằm trên Drive dự án: `resources/papers/<Paper>/`
(không đưa PDF lên repo). Cập nhật Drive: `bash scripts/drive_publish_data.sh resources`.

| Paper | Venue | arXiv | Code | Input |
|---|---|---|---|---|
| **ArtGS**: Building Interactable Replicas of Complex Articulated Objects via Gaussian Splatting | ICLR 2025 | [2502.19459](https://arxiv.org/abs/2502.19459) | [YuLiu-LY/ArtGS](https://github.com/YuLiu-LY/ArtGS) | RGB(-D), 2 trạng thái, nhiều part |
| **REArtGS**: Reconstructing and Generating Articulated Objects via 3D Gaussian Splatting with Geometric and Motion Constraints | NeurIPS 2025 | [2503.06677](https://arxiv.org/abs/2503.06677) | [wd-ustc-cs/REArtGS](https://github.com/wd-ustc-cs/REArtGS) | RGB, 2 trạng thái, 2 part |
| **REArtGS++**: Generalizable Articulation Reconstruction with Temporal Geometry Constraint via Planar Gaussian Splatting | CVPR 2026 | [2511.17059](https://arxiv.org/abs/2511.17059) | [wd-ustc-cs/REArtGS2](https://github.com/wd-ustc-cs/REArtGS2) | RGB, 2 trạng thái, nhiều part |
| **Rest2Art**: Articulated Object Reconstruction from Rest-State Observation | ECCV 2026 | [2607.27749](https://arxiv.org/abs/2607.27749) | chưa công bố | 1 trạng thái nghỉ (đóng) |

## Tóm tắt

- **ArtGS**: canonical Gaussians + dual-quaternion slot skinning để căn chỉnh 2 trạng thái; module
  riêng dự đoán loại khớp (xoay/tịnh tiến); khởi tạo part bằng spectral clustering; benchmark
  ArtGS-Multi (5 vật, 3-6 part).
- **REArtGS**: thêm SDF guidance không lệch để regularize opacity field (mesh sạch hơn); trường biến
  dạng ràng buộc bởi cấu trúc động học, sinh mesh ở trạng thái bất kỳ t ∈ [0,1]; vẫn cần prior loại khớp.
- **REArtGS++**: bỏ prior loại khớp bằng screw motion SE(3) học trực tiếp; planar Gaussians cho
  normal/depth chính xác; ràng buộc hình học theo thời gian (Taylor bậc 1 theo t); segmentation mềm
  (Mahalanobis + MLP) và voting cho vùng biên.
- **Rest2Art**: chỉ dùng **một** trạng thái đóng, không quan sát chuyển động. Mesh tường minh làm biểu
  diễn trung gian để hợp nhất output của VLM và model segmentation thành cấu trúc part nhất quán; tham
  số khớp ước lượng bằng giả thuyết chuyển động từ video diffusion, kiểm chứng bằng nhất quán hình học.
  Đánh giá trên ACD, so với PARIS / ArtGS / REArtGS / REArtGS++.

## Lưu ý khi so sánh số liệu

REArtGS và REArtGS++ tắt depth supervision của ArtGS khi so sánh, nên số của ArtGS trong hai paper đó
thấp hơn paper gốc. Trong repo này baseline luôn là **checkpoint chính thức chạy lại trong môi trường
của mình** (`scripts/jobs/eval_official.sh`), không lấy số từ paper.
