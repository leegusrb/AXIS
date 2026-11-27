import 'dart:typed_data';

import 'package:axis_app/camera/frame_encoder.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;

/// 좌상단 10x10 빨강, 나머지 파랑인 BGRA 프레임. [stride]가 크면 행 끝에 패딩.
FrameData bgraFrame({int w = 40, int h = 20, int stride = 0, int rot = 0}) {
  final rowBytes = stride == 0 ? w * 4 : stride;
  final b = Uint8List(rowBytes * h);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      final red = x < 10 && y < 10;
      final i = y * rowBytes + x * 4;
      b[i] = red ? 0 : 255; // B
      b[i + 1] = 0; // G
      b[i + 2] = red ? 255 : 0; // R
      b[i + 3] = 255;
    }
    // 패딩 영역은 쓰레기 값
    for (var x = w * 4; x < rowBytes; x++) {
      b[y * rowBytes + x] = 77;
    }
  }
  return FrameData(
    width: w,
    height: h,
    format: FrameFormat.bgra8888,
    planes: [FramePlane(b, rowBytes)],
    rotationDegrees: rot,
  );
}

bool isRed(img.Pixel p) => p.r > 200 && p.g < 80 && p.b < 80;
bool isBlue(img.Pixel p) => p.b > 200 && p.r < 80 && p.g < 80;

void main() {
  test('encoder rotates 90 so top-left red pixel ends at top-right', () {
    final out = img.decodeJpg(encodeFrame(bgraFrame(rot: 90)))!;
    expect(out.width, 20);
    expect(out.height, 40);
    expect(isRed(out.getPixel(17, 2)), isTrue); // top-right
    expect(isBlue(out.getPixel(2, 2)), isTrue); // top-left
    expect(isBlue(out.getPixel(17, 37)), isTrue); // bottom-right
  });

  test('encoder rotates 270 to bottom-left and 180 to bottom-right', () {
    final r270 = img.decodeJpg(encodeFrame(bgraFrame(rot: 270)))!;
    expect(isRed(r270.getPixel(2, 37)), isTrue);
    final r180 = img.decodeJpg(encodeFrame(bgraFrame(rot: 180)))!;
    expect(isRed(r180.getPixel(37, 17)), isTrue);
  });

  test('encoder keeps rotation 0 orientation', () {
    final out = img.decodeJpg(encodeFrame(bgraFrame()))!;
    expect(isRed(out.getPixel(2, 2)), isTrue);
    expect(isBlue(out.getPixel(37, 17)), isTrue);
  });

  test('encoder scales long side to 320 after rotation', () {
    final f = bgraFrame(w: 640, h: 480, rot: 90);
    final out = img.decodeJpg(encodeFrame(f))!;
    expect(out.height, 320);
    expect(out.width, 240);
  });

  test('encoder ignores bytesPerRow padding', () {
    final plain = img.decodeJpg(encodeFrame(bgraFrame(rot: 90)))!;
    final padded = img.decodeJpg(encodeFrame(bgraFrame(stride: 176, rot: 90)))!;
    expect(padded.width, plain.width);
    expect(padded.height, plain.height);
    expect(padded.getBytes(), plain.getBytes());
  });

  test('encoder converts yuv420 gray frame without error', () {
    const w = 64, h = 48;
    final y = Uint8List(w * h)..fillRange(0, w * h, 128);
    final u = Uint8List(w * h ~/ 2)..fillRange(0, w * h ~/ 2, 128);
    final f = FrameData(
      width: w,
      height: h,
      format: FrameFormat.yuv420,
      planes: [
        FramePlane(y, w),
        FramePlane(u, w, 2), // 인터리브된 UV, 픽셀 간격 2
        FramePlane(u, w, 2),
      ],
      rotationDegrees: 90,
    );
    final out = img.decodeJpg(encodeFrame(f))!;
    expect(out.width, h);
    expect(out.height, w);
    final p = out.getPixel(10, 10);
    for (final c in [p.r, p.g, p.b]) {
      expect(c, closeTo(128, 4));
    }
  });

  test('yuv420 colors follow BT.601', () {
    const w = 8, h = 8;
    final y = Uint8List(w * h)..fillRange(0, w * h, 82);
    final u = Uint8List(16)..fillRange(0, 16, 90);
    final v = Uint8List(16)..fillRange(0, 16, 240);
    final f = FrameData(
      width: w,
      height: h,
      format: FrameFormat.yuv420,
      planes: [FramePlane(y, w), FramePlane(u, 4, 1), FramePlane(v, 4, 1)],
      rotationDegrees: 0,
    );
    final p = img.decodeJpg(encodeFrame(f))!.getPixel(4, 4);
    expect(p.r, greaterThan(220));
    expect(p.g, lessThan(40));
    expect(p.b, lessThan(40));
  });
}
