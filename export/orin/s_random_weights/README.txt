MOTIP2 scale S, 31 keypoints, ONE graph (detector + trajectory adapter + ID decoder) -- RANDOM WEIGHTS (torch.manual_seed(0)),
NOT a trained model. Config configs/motip2/s_vehicles_kp.yaml, exported at 800x1440.
Purpose: test the Orin / TensorRT toolchain only (ONNX parser, operator support, engine build, FP16/INT8 builder, latency, memory).
Detections and tracks are meaningless. Replace with the real export:
  CKPT_S=outputs/s_vehicles_kp/checkpoint_9.pth SIZES=S bash tools/export_orin.sh
Inputs : x (1,3,800,1440) float32 RGB [0,1] (1080p frame resized to 1422x800, zero-padded); det_thresh (1,) float32;
         traj_feat/traj_pos (T,N,256), traj_ids/traj_times (T,N) int64, traj_masks (T,N) bool   (T in [2,64], N in [1,300])
Outputs: logits, boxes, emb, pos, kpts, kpt_vis, id_logits   (see docs/DEPLOY_ORIN.md section 3a; operators in export_meta.json)
Verified with onnxruntime: detector vs PyTorch 2.4e-06; selected id_logits vs the two-graph pipeline 3.1e-06. No TensorRT engine was built.
