import 'package:flutter/material.dart';

/// 작가 카드의 구도 그림: 세로 화면 틀, 삼분할 점선, 피사체 박스(0–1 정규화).
class CompositionDiagram extends StatelessWidget {
  const CompositionDiagram({super.key, required this.box, this.aspect = 3 / 4});
  final Rect box;
  final double aspect;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return AspectRatio(
      aspectRatio: aspect,
      child: CustomPaint(
        painter: _DiagramPainter(box, scheme.primary, scheme.outline),
      ),
    );
  }
}

class _DiagramPainter extends CustomPainter {
  _DiagramPainter(this.box, this.subject, this.line);
  final Rect box;
  final Color subject;
  final Color line;

  @override
  void paint(Canvas canvas, Size size) {
    final frame = Offset.zero & size;
    canvas.drawRect(frame, Paint()..color = line.withValues(alpha: 0.08));
    final thin = Paint()
      ..color = line.withValues(alpha: 0.5)
      ..strokeWidth = 1;
    for (final f in [1 / 3, 2 / 3]) {
      canvas.drawLine(
        Offset(size.width * f, 0),
        Offset(size.width * f, size.height),
        thin,
      );
      canvas.drawLine(
        Offset(0, size.height * f),
        Offset(size.width, size.height * f),
        thin,
      );
    }
    final r = Rect.fromLTWH(
      box.left * size.width,
      box.top * size.height,
      box.width * size.width,
      box.height * size.height,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(r, const Radius.circular(3)),
      Paint()..color = subject.withValues(alpha: 0.6),
    );
    canvas.drawRect(
      frame,
      Paint()
        ..color = line
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5,
    );
  }

  @override
  bool shouldRepaint(_DiagramPainter old) =>
      old.box != box || old.subject != subject || old.line != line;
}
