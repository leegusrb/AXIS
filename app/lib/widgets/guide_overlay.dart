import 'dart:convert';

import 'package:flutter/material.dart';

import '../api/models.dart';

/// 0–1 정규화 박스를 [size] 크기 사각형의 픽셀 좌표로 바꾼다.
Rect toCanvasRect(Rect normalized, Size size) => Rect.fromLTWH(
  normalized.left * size.width,
  normalized.top * size.height,
  normalized.width * size.width,
  normalized.height * size.height,
);

/// 인물 박스(얇은 흰 선)와 권장 위치(점선)만 그린다. `result == null`이면 그리지 않는다.
class GuideOverlayPainter extends CustomPainter {
  const GuideOverlayPainter(this.result, {this.showTarget = true});
  final GuideResult? result;
  final bool showTarget;

  @override
  void paint(Canvas canvas, Size size) {
    final r = result;
    if (r == null) return;
    final subject = r.subjectBox;
    if (subject != null) {
      canvas.drawRect(
        toCanvasRect(subject, size),
        Paint()
          ..color = Colors.white
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1,
      );
    }
    final target = r.targetBox;
    if (showTarget && target != null) {
      _dashedRect(
        canvas,
        toCanvasRect(target, size),
        Paint()
          ..color = Colors.amberAccent
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2,
      );
    }
  }

  void _dashedRect(Canvas canvas, Rect rect, Paint paint) {
    final path = Path()..addRect(rect);
    for (final metric in path.computeMetrics()) {
      var d = 0.0;
      while (d < metric.length) {
        canvas.drawPath(metric.extractPath(d, d + 8), paint);
        d += 14;
      }
    }
  }

  @override
  bool shouldRepaint(GuideOverlayPainter old) =>
      old.result != result || old.showTarget != showTarget;
}

IconData? guideIcon(String direction) => switch (direction) {
  'LEFT' => Icons.arrow_back,
  'RIGHT' => Icons.arrow_forward,
  'UP' => Icons.arrow_upward,
  'DOWN' => Icons.arrow_downward,
  'CLOSER' => Icons.zoom_in,
  'FARTHER' => Icons.zoom_out,
  'OK' => Icons.check_circle,
  'NO_SUBJECT' => Icons.person_search,
  _ => null,
};

/// 프리뷰 위에 얹는 일반 위젯 오버레이: 큰 아이콘, 문구, 점수 막대.
const rotateMessages = {
  'CW': '휴대폰을 시계 방향으로 살짝 돌리세요',
  'CCW': '휴대폰을 반시계 방향으로 살짝 돌리세요',
};

class GuideHud extends StatelessWidget {
  const GuideHud({super.key, required this.result, this.rotate});
  final GuideResult? result;
  final String? rotate; // 보조 안내: 위쪽 가운데에 작게

  @override
  Widget build(BuildContext context) {
    final r = result;
    if (r == null) return const SizedBox.shrink();
    final icon = guideIcon(r.direction);
    final ok = r.direction == 'OK';
    final rot = rotate;
    return Stack(
      fit: StackFit.expand,
      children: [
        if (rot != null && rotateMessages.containsKey(rot))
          Positioned(
            top: 12,
            left: 16,
            right: 16,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(
                  rot == 'CW' ? Icons.rotate_right : Icons.rotate_left,
                  color: Colors.amberAccent,
                  size: 36,
                  shadows: const [Shadow(blurRadius: 6, color: Colors.black87)],
                ),
                const SizedBox(width: 6),
                Flexible(
                  child: Text(
                    rotateMessages[rot]!,
                    style: const TextStyle(
                      color: Colors.amberAccent,
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      shadows: [Shadow(blurRadius: 6, color: Colors.black87)],
                    ),
                  ),
                ),
              ],
            ),
          ),
        if (icon != null)
          Center(
            child: Icon(
              icon,
              size: r.magnitude == 'LARGE' ? 140 : 80,
              color: ok ? Colors.greenAccent : Colors.white,
              shadows: const [Shadow(blurRadius: 8, color: Colors.black54)],
            ),
          ),
        Positioned(
          left: 16,
          right: 16,
          bottom: 12,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                r.message,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  shadows: [Shadow(blurRadius: 6, color: Colors.black87)],
                ),
              ),
              const SizedBox(height: 8),
              ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value: r.score.clamp(0.0, 1.0),
                  minHeight: 8,
                  backgroundColor: Colors.black38,
                  color: ok ? Colors.greenAccent : Colors.white,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

/// 서버가 보낸 깊이 지도(흑백, 가까울수록 밝게)를 작은 창으로 보여 준다.
class DepthPreview extends StatelessWidget {
  const DepthPreview({super.key, required this.base64Png, this.width = 96});
  final String? base64Png;
  final double width;

  @override
  Widget build(BuildContext context) {
    final b = base64Png;
    if (b == null) return const SizedBox.shrink();
    return Container(
      width: width,
      decoration: BoxDecoration(border: Border.all(color: Colors.white70)),
      child: Stack(
        children: [
          Image.memory(
            base64Decode(b),
            width: width,
            fit: BoxFit.fitWidth,
            gaplessPlayback: true,
          ),
          const Positioned(
            left: 4,
            top: 2,
            child: Text(
              '깊이',
              style: TextStyle(color: Colors.white, fontSize: 11),
            ),
          ),
        ],
      ),
    );
  }
}
