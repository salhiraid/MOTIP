MOTIP2 scale S, 31 keypoints (configs/motip2/s_vehicles_kp.yaml) -- RANDOM WEIGHTS (torch.manual_seed(0)), NOT a trained model.
Purpose: test the Orin / TensorRT toolchain only (ONNX parser, operator support, engine build, FP16/INT8 builder, latency,
memory). Its detections and tracks are meaningless. Replace with the real export:
  CKPT_S=outputs/s_vehicles_kp/checkpoint_9.pth SIZES=S bash tools/export_orin.sh
Detector input: (1,3,800,1440) float32 RGB in [0,1] (1080p frame resized to 1422x800, zero-padded). Outputs/inputs/ops/opset: export_meta.json.
Verified against PyTorch with onnxruntime: detector max diff 2.4e-06, ID decoder 2.9e-06. TensorRT engines were NOT built.
