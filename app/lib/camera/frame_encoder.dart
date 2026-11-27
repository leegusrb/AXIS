import 'dart:typed_data';

import 'package:image/image.dart' as img;

import '../config.dart';

enum FrameFormat { bgra8888, yuv420 }

/// isolate(`compute`)로 넘길 수 있는 순수 데이터.
class FramePlane {
  const FramePlane(this.bytes, this.bytesPerRow, [this.bytesPerPixel]);
  final Uint8List bytes;
  final int bytesPerRow;
  final int? bytesPerPixel;
}

class FrameData {
  const FrameData({
    required this.width,
    required this.height,
    required this.format,
    required this.planes,
    required this.rotationDegrees,
  });
  final int width;
  final int height;
  final FrameFormat format;
  final List<FramePlane> planes;
  final int rotationDegrees;
}

int _clamp255(double v) => v < 0 ? 0 : (v > 255 ? 255 : v.round());

/// 긴 변을 [frameLongSide]로 줄이고(작은 프레임은 키우지 않는다),
/// 시계 방향으로 `rotationDegrees`(0/90/180/270)만큼 돌려 JPEG로 만든다.
Uint8List encodeFrame(FrameData f) =>
    Uint8List.fromList(img.encodeJpg(frameToRgb(f), quality: jpegQuality));

/// 카메라 프레임을 긴 변 [frameLongSide] 이하의 RGB 이미지로 바꾸고 화면 방향으로 돌린다.
img.Image frameToRgb(FrameData f) {
  final longSide = f.width > f.height ? f.width : f.height;
  final scale = longSide > frameLongSide ? frameLongSide / longSide : 1.0;
  final tw = (f.width * scale).round().clamp(1, f.width);
  final th = (f.height * scale).round().clamp(1, f.height);

  final out = img.Image(width: tw, height: th, numChannels: 3);
  for (var y = 0; y < th; y++) {
    final sy = (y * f.height ~/ th).clamp(0, f.height - 1);
    for (var x = 0; x < tw; x++) {
      final sx = (x * f.width ~/ tw).clamp(0, f.width - 1);
      switch (f.format) {
        case FrameFormat.bgra8888:
          final p = f.planes[0];
          final i = sy * p.bytesPerRow + sx * 4;
          out.setPixelRgb(x, y, p.bytes[i + 2], p.bytes[i + 1], p.bytes[i]);
        case FrameFormat.yuv420:
          final yp = f.planes[0], up = f.planes[1], vp = f.planes[2];
          final yy = yp.bytes[sy * yp.bytesPerRow + sx].toDouble();
          final uStep = up.bytesPerPixel ?? 1;
          final vStep = vp.bytesPerPixel ?? 1;
          final u =
              up.bytes[(sy >> 1) * up.bytesPerRow + (sx >> 1) * uStep] - 128;
          final v =
              vp.bytes[(sy >> 1) * vp.bytesPerRow + (sx >> 1) * vStep] - 128;
          out.setPixelRgb(
            x,
            y,
            _clamp255(yy + 1.402 * v),
            _clamp255(yy - 0.344136 * u - 0.714136 * v),
            _clamp255(yy + 1.772 * u),
          );
      }
    }
  }

  return f.rotationDegrees % 360 == 0
      ? out
      : img.copyRotate(out, angle: f.rotationDegrees % 360);
}
