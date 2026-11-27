import 'dart:typed_data';

import 'package:axis_app/camera/frame_encoder.dart';
import 'package:axis_app/local/preprocess.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;

double px(Float32List t, int c, int y, int x, {int size = 320}) =>
    t[c * size * size + y * size + x];

void main() {
  test('toModelInput normalizes and letterboxes', () {
    final im = img.Image(width: 640, height: 320, numChannels: 3)
      ..clear(img.ColorRgb8(255, 0, 0));
    final (t, lb) = toModelInput(im);
    expect(t.length, 3 * 320 * 320);
    expect(lb.scale, 0.5);
    expect(lb.padY, 80);
    expect(lb.padX, 0);
    expect(px(t, 0, 160, 160), closeTo(1.0, 1e-6)); // 빨강
    expect(px(t, 1, 160, 160), closeTo(0.0, 1e-6));
    expect(px(t, 0, 10, 160), closeTo(114 / 255, 1e-6)); // 위쪽 padding
  });

  test('preprocess rotates then letterboxes', () {
    // BGRA 4×2: 왼쪽 절반 빨강, 오른쪽 절반 파랑. 90° 돌리면 2×4 세로, 위가 빨강.
    final bytes = Uint8List(4 * 2 * 4);
    for (var y = 0; y < 2; y++) {
      for (var x = 0; x < 4; x++) {
        final i = (y * 4 + x) * 4;
        if (x < 2) {
          bytes[i + 2] = 255; // R
        } else {
          bytes[i] = 255; // B
        }
        bytes[i + 3] = 255;
      }
    }
    final f = FrameData(
      width: 4,
      height: 2,
      format: FrameFormat.bgra8888,
      planes: [FramePlane(bytes, 16, 4)],
      rotationDegrees: 90,
    );
    final input = prepareLocal(f);
    expect(input.lb.srcW, 2);
    expect(input.lb.srcH, 4);
    expect(input.lb.padX, 80); // 세로 이미지가 가운데, 좌우 padding
    final t = input.input;
    expect(px(t, 0, 40, 160), closeTo(1.0, 1e-6)); // 위: 빨강
    expect(px(t, 2, 280, 160), closeTo(1.0, 1e-6)); // 아래: 파랑
    expect(px(t, 0, 160, 10), closeTo(114 / 255, 1e-6));
  });
}
