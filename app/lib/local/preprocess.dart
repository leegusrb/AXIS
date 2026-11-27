import 'dart:typed_data';

import 'package:image/image.dart' as img;

import '../camera/frame_encoder.dart';
import 'yolo_decode.dart';

const modelInputSize = 320;
const _pad = 114 / 255;

/// 가운데 정렬 letterbox(회색 114) 후 0–1 float NCHW [1,3,size,size].
(Float32List, Letterbox) toModelInput(
  img.Image rgb, {
  int size = modelInputSize,
}) {
  final s = (size / rgb.width) < (size / rgb.height)
      ? size / rgb.width
      : size / rgb.height;
  final nw = (rgb.width * s).round().clamp(1, size),
      nh = (rgb.height * s).round().clamp(1, size);
  final px = (size - nw) ~/ 2, py = (size - nh) ~/ 2;
  final r = img.copyResize(
    rgb,
    width: nw,
    height: nh,
    interpolation: img.Interpolation.linear,
  );
  final bytes = r.getBytes(order: img.ChannelOrder.rgb);
  final plane = size * size;
  final t = Float32List(3 * plane)..fillRange(0, 3 * plane, _pad);
  for (var y = 0; y < nh; y++) {
    for (var x = 0; x < nw; x++) {
      final i = (y * nw + x) * 3, o = (y + py) * size + (x + px);
      t[o] = bytes[i] / 255;
      t[plane + o] = bytes[i + 1] / 255;
      t[2 * plane + o] = bytes[i + 2] / 255;
    }
  }
  final lb = Letterbox(
    scale: s,
    padX: px.toDouble(),
    padY: py.toDouble(),
    srcW: rgb.width,
    srcH: rgb.height,
    size: size,
  );
  return (t, lb);
}

class LocalInput {
  const LocalInput(this.input, this.lb);
  final Float32List input;
  final Letterbox lb;
}

/// `compute`로 isolate에서 돌리는 전처리: 카메라 프레임 → 모델 입력.
LocalInput prepareLocal(FrameData f) {
  final (t, lb) = toModelInput(frameToRgb(f));
  return LocalInput(t, lb);
}
