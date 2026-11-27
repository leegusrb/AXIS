import 'package:camera/camera.dart';

import 'frame_encoder.dart';

/// camera 플러그인 타입을 순수 데이터로 바꾼다(실기기에서만 검증).
FrameData frameDataFromCameraImage(CameraImage img, int rotationDegrees) {
  return FrameData(
    width: img.width,
    height: img.height,
    format: img.format.group == ImageFormatGroup.bgra8888
        ? FrameFormat.bgra8888
        : FrameFormat.yuv420,
    planes: [
      for (final p in img.planes)
        FramePlane(p.bytes, p.bytesPerRow, p.bytesPerPixel),
    ],
    rotationDegrees: rotationDegrees,
  );
}
