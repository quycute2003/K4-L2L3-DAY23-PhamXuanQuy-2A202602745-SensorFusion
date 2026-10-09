# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Phạm Xuân Quý
- MSSV: 2A202602745
- Email: 26ai.quypx@vinuni.edu.vn
- Link repo (fork): https://github.com/quycute2003/K4-L2L3-DAY23-PhamXuanQuy-2A202602745-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): Lấy hash 40 ký tự của commit cuối cùng chứa báo cáo này bằng `git rev-parse HEAD` sau CP6; dùng chính hash đó khi nộp LMS.

## Tóm tắt kết quả

- `fusion_mode`: `compare`; `frames`: `[0, 198]` (199 frame mỗi mode); `seed`: `0`.
- `segment`: `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`.
- `detection.precision`: `0.9700934579439252`; `detection.recall`: `0.7004048582995951`; `detection.tp/fp/fn`: `519/16/222`.

| Chỉ số tracking | LiDAR | LiDAR + camera (fused) |
|---|---:|---:|
| RMSE vị trí 3D (m) | 0.15032268781360136 | 0.1358667883353908 |
| `matches` | 502 | 502 |
| `sum_sq_err` (m²) | 11.343649056695737 | 9.266811654632088 |
| `ghost_track_frames` | 0 | 0 |
| `missed_gt_frames` | 239 | 239 |
| `mean_confirmed_tracks` | 2.522613065326633 | 2.522613065326633 |
| `precision_track` | 1.0 | 1.0 |
| `coverage = matches / det_tp` | 0.9672447013487476 | 0.9672447013487476 |

- Giải thích khác biệt hai mode: Camera làm giảm RMSE tổng thể `0.014455899478210577 m` (khoảng 9.62%) và giảm tổng bình phương sai số trên cùng 502 cặp ghép. Matches, ghost và miss đều giữ nguyên, nên mức cải thiện RMSE không đi kèm việc đánh giá ít cặp hơn hoặc bỏ thêm track. Tổng thể phù hợp với vai trò camera tinh chỉnh state sau lượt LiDAR; log tổng hợp không chứa ID để khẳng định từng cặp GT–track giống nhau giữa hai mode.
- Camera không cải thiện mọi frame: frame 93 có `sum_sq_err` giảm từ `0.042219327427724024` xuống `0.032096904387808156` trên 2 matches; frame 4 tăng từ `0.011281520740472877` lên `0.03198731678441609` trên 2 matches. Đo camera có nhiễu và mô hình pinhole phi tuyến nên cần đánh giá cả segment, không chọn một frame đẹp.
- Tổng có 741 GT xe hợp lệ theo frame; detector bỏ sót 222 và confirmed tracker bỏ sót 239, chênh lệch 17. Giai đoạn chờ xác nhận góp phần tạo miss: frame 0 detector có 2 TP nhưng chưa có confirmed track; đến frame 4 đã có 2 confirmed matches. Không có ghost trong lần chạy này không đồng nghĩa hệ thống luôn không báo nhầm.
- Đối chiếu rubric: cả hai RMSE ≤ 0.45 m, precision track ≥ 0.75, coverage ≥ 0.70; `rmse_fused − rmse_lidar = -0.014455899478210577 m ≤ 0.05 m`, nên `q_lidar = q_fused = 1`. Metrics/log đáp ứng phần A (15 điểm) và các ngưỡng B tự động (45 điểm), với điều kiện E–H qua bộ test gốc khi chấm. Phần thủ công và báo cáo do giảng viên đánh giá.
- Đã kiểm tra đủ 398 record, đúng một record mỗi `(mode, frame)`; mọi invariant đúng và tổng/trung bình tái tạo chính xác metrics. Bốn file per-mode cũng khớp file compare.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

Lần chạy này dùng Windows, Python 3.12.14, NumPy 2.5.3, PyTorch 2.14.1 CPU; đặt `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`. Cấu hình local dùng `frame_start: 0`, `frame_end: 198`. Trên Windows, bật `PYTHONUTF8=1` để bộ test và công cụ đọc/ghi tiếng Việt đúng encoding.

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?

   LiDAR đo tâm hộp 3D trong hệ cảm biến, nên `z` có kích thước 3×1, đơn vị mét; `R` là 3×3 với các phương sai `sigma_lidar_x/y/z²`. Mô hình đo LiDAR tuyến tính: đổi vị trí từ hệ xe sang hệ cảm biến; `H` có kích thước 3×6 và ba cột vận tốc bằng 0.

   Camera đo tâm hộp ảnh `[u, v]`, nên `z` là 2×1, đơn vị pixel; `R = diag(sigma_cam_i², sigma_cam_j²)` là 2×2. Sau phép đổi hệ `p_s = R_rotation p_vehicle + t`, mô hình pinhole là `u = c_i − f_i y_s/x_s`, `v = c_j − f_j z_s/x_s`. Mô hình này phi tuyến nên `H` là Jacobian 2×6 do platform cung cấp. Hai sensor dùng chung state 6D nhưng khác kích thước và đơn vị đo; không cộng trực tiếp covariance mét và pixel. Xem [camera_fusion.py](workspace/camera_fusion.py), [kalman.py](workspace/kalman.py) và [Sensor](../platform/fusion_lab/tracking/sensors.py).

2. Vì sao cần gating Mahalanobis trước khi gán?

   Innovation `γ = z − h(x)` và covariance `S = HPHᵀ + R` tạo khoảng cách bình phương `d² = γᵀS⁻¹γ`. Đây là sai lệch đã được chuẩn hóa theo độ bất định: cùng một residual, track có covariance lớn có thể được chấp nhận hơn track có covariance nhỏ. Euclidean chỉ nhìn khoảng cách, không xét độ bất định này.

   Trong [association.py](workspace/association.py), cặp chỉ được đưa vào gán greedy nếu track nằm trong FOV và `d² < chi2.ppf(gating_threshold, dim_meas)`; xác suất gate là 0.995, số chiều là 3 cho LiDAR và 2 cho camera. Cặp bị loại mang cost `inf`. Gate giảm khả năng ghép một đo xa hoặc khác xe chỉ vì đó là đo gần nhất còn lại. Kiểm tra FOV xảy ra trước projection/Mahalanobis để không chiếu điểm camera phía sau hoặc có độ sâu không hợp lệ.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.

   Đây là **track-then-fuse**: một danh sách track, predict một lần mỗi frame, gán và update LiDAR, quản lý vòng đời LiDAR, rồi gán và update camera trên chính danh sách đó khi chạy fused. Runner thể hiện thứ tự này tại [run_lab.py](../platform/fusion_lab/scripts/run_lab.py), các lệnh `KF.predict`, `associate_and_update(..., lidar_sensor)` rồi `associate_and_update(..., camera_sensor)`.

   [grade_run.log](artifacts/grade_run.log) có record `lidar` và `fused` cho mỗi frame, với detection dùng chung; `matches`, `sum_sq_err`, `ghosts`, `misses` là kết quả đánh giá sau lượt tracking của frame. Log không ghi riêng từng predict/update nên thứ tự sensor cần đối chiếu với runner, không suy ra chỉ từ nhãn mode. Waymo được xử lý đồng bộ theo frame, `Measurement.t = frame_index × dt`; vì hai sensor cùng thời điểm logic, không predict thêm giữa hai lượt update. Một hệ thống async thực tế phải predict tới từng thời điểm đo.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?

   Extrinsic sai làm vị trí 3D được đổi sang hệ camera sai, vì vậy `h(x)` lệch so với pixel đo. Innovation có thể có độ lệch có hệ thống, cùng dấu hoặc phụ thuộc vị trí/khoảng cách xe, thay vì dao động quanh 0. Nếu `γᵀS⁻¹γ` vượt gate, đo camera bị loại và track tiếp tục dựa vào LiDAR. Nếu sai lệch vẫn nằm trong gate, camera update có thể kéo vị trí/vận tốc sai và tăng RMSE fused. Sai calibration còn có thể làm FOV sai, gây bỏ đo hoặc ghép nhầm. Đây là giải thích từ mô hình; bài này chưa thực hiện thí nghiệm làm lệch calibration, nên không có số liệu thực nghiệm cho trường hợp đó.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.

   Khi `meas_list` rỗng, không thể lấy sensor từ đo đầu tiên. Tham số `sensor` tường minh cho phép [associate_and_update](workspace/association.py) vẫn gọi `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)` đúng lượt. Lượt LiDAR rỗng phải trừ score của track chưa ghép nhưng trong FOV và kiểm tra xóa; lượt camera rỗng không được coi là bằng chứng xe biến mất.

   LiDAR cung cấp vị trí 3D để khởi tạo track và là nguồn quyết định tồn tại trong thiết kế lab. Camera chỉ tinh chỉnh state/covariance qua EKF: camera hit không tăng score, camera miss không giảm score, camera không tạo/xóa track. [TrackManager](../platform/fusion_lab/tracking/manager.py) kiểm tra `sensor.name` khi xử lý hit và quản lý vòng đời; các regression test kiểm tra cả camera rỗng, đo không ghép được và camera hit.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.

   Theo [track_management.py](workspace/track_management.py), track LiDAR mới có vị trí đổi về hệ xe, vận tốc khởi tạo 0, covariance vị trí `R_rotation R_measurement R_rotationᵀ`, covariance vận tốc theo `sigma_p44/55/66²`, state `initialized` và score `1/window = 1/6`. LiDAR hit tăng score thêm `1/6`, tối đa 1; hit khi chưa đạt xác nhận chuyển thành `tentative`. Track được xác nhận khi **score > 0.8**, không phải khi bằng 0.8.

   Một miss LiDAR trong FOV giảm score `1/6` nhưng giữ state `confirmed` nếu track đã được xác nhận. Ví dụ score 1 sau một miss còn 5/6, vẫn cao hơn ngưỡng xóa. Xóa nếu **bất kỳ** điều kiện nào đúng: `P[0,0] > 9` hoặc `P[1,1] > 9`; track confirmed có score **< 0.6**; track chưa confirmed có score **≤ 0**. Bằng ngưỡng covariance 9 hoặc score confirmed 0.6 chưa bị xóa. Track ngoài FOV không bị trừ score vì miss; camera không kích hoạt các quyết định này.

## Bonus (không bắt buộc)

- Không.

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): ChatGPT (Codex).
- Dùng cho phần nào (hàm, câu hỏi, debug): Codex thực hiện CP0, implement toàn bộ hàm Part E–H, chạy test, chạy Waymo compare, kiểm tra artifacts, soạn tóm tắt số liệu và cả 6 câu giải thích trong báo cáo. Codex cũng tạo commit theo checkpoint và chuẩn bị nộp repo.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Các kiểm tra do Codex thực hiện: `pytest student/tests -q` có 128 passed, không failed/error/xfailed; kiểm tra số học khởi tạo track với LiDAR có rotation/translation; chạy `fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0` trên đủ frame 0–198; tái tạo metrics từ 398 record bằng validator platform; đối chiếu toàn bộ file per-mode. Không sửa platform, tests hoặc số liệu artifacts. EKF dùng `solve` thay inverse trực tiếp và Joseph form cho covariance, tương đương công thức cập nhật nhưng ổn định hơn về số học.

Theo [RULES.md §2](../RULES.md), học viên cần tự đọc, kiểm tra và giải thích được code/câu trả lời trước khi nộp; chưa ghi nhận bước tự kiểm tra này của học viên. Mẫu báo cáo ghi phần giải thích là “tự viết”; các câu trả lời hiện tại do Codex soạn và đã được khai báo ở trên. Test và metrics không thay thế việc học viên hiểu bài để vấn đáp. Không khai báo AI bị trừ 10 điểm; phần không giải thích được bị tính 0 điểm. Tài liệu không quy định trần điểm tự động chỉ vì dùng AI.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không sửa
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [ ] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
