import 'dart:async';
import 'dart:io' show Platform;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../api/axis_api.dart';
import '../api/models.dart';
import '../camera/camera_frame.dart';
import '../camera/frame_encoder.dart';
import '../config.dart';
import '../guide/guide_controller.dart';
import '../local/local_analyzer.dart';
import '../state/app_state.dart';
import '../widgets/guide_overlay.dart';
import 'errors.dart';
import 'result_screen.dart';
import 'settings_screen.dart' show scorerLabel;

/// 서버로 보내는 JPEG가 프리뷰(세로)와 같은 방향이 되도록 돌릴 각도.
/// 실기기에서 박스가 어긋나면 이 함수만 고친다.
int frameRotationFor(CameraDescription d) =>
    Platform.isAndroid ? d.sensorOrientation : 0;

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  CameraController? _cam;
  CameraDescription? _desc;
  GuideController<FrameData>? _guide;
  AppState? _state;
  String? _error;
  bool _starting = false;
  bool _encoding = false;
  bool _saving = false;
  bool _guideOff = false;
  bool _depthOn = false;
  bool _local = false; // 폰에서 분석
  bool _inBackground = false;
  Future<void>? _releasing;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (_state != null) return;
    final state = AppScope.of(context);
    _state = state;
    final style = state.selectedStyle;
    if (style == null) {
      final messenger = ScaffoldMessenger.of(context);
      final navigator = Navigator.of(context);
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (navigator.canPop()) navigator.pop();
        messenger.showSnackBar(const SnackBar(content: Text('먼저 스타일을 골라 주세요')));
      });
      return;
    }
    _local = state.analysis == 'local';
    _guide = GuideController<FrameData>(
      _local
          ? (f) async => (await LocalAnalyzer.load()).analyze(f, style)
          : (f) async => state.api.analyze(
              await compute(encodeFrame, f),
              style.id,
              scorer: state.scorer,
              depthPreview: _depthOn,
            ),
    );
    _initCamera();
  }

  Future<void> _initCamera({bool retry = true}) async {
    if (_starting || _cam != null) return;
    _starting = true;
    try {
      await _releasing;
      if (!mounted || _inBackground) return;
      final cams = await availableCameras();
      final back = cams
          .where((c) => c.lensDirection == CameraLensDirection.back)
          .toList();
      if (back.isEmpty) {
        _fail('후면 카메라를 찾을 수 없어요');
        return;
      }
      final desc = back.first;
      final cam = CameraController(
        desc,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: Platform.isIOS
            ? ImageFormatGroup.bgra8888
            : ImageFormatGroup.yuv420,
      );
      await cam.initialize();
      // 기기를 가로로 기울여도 스트림 프레임이 세로로 유지되게 고정한다.
      await cam.lockCaptureOrientation(DeviceOrientation.portraitUp);
      if (!mounted || _inBackground) {
        await _disposeTracked(cam);
        return;
      }
      _desc = desc;
      setState(() {
        _cam = cam;
        _error = null;
      });
      await _startStream();
    } on CameraException catch (e) {
      _fail(
        e.code.contains('Denied')
            ? '카메라 권한이 필요해요. 설정에서 허용해 주세요'
            : '카메라를 사용할 수 없어요',
      );
    } catch (_) {
      _fail('카메라를 사용할 수 없어요');
    } finally {
      _starting = false;
      // 중단 중에 resumed가 와서 무시됐다면 한 번 다시 시작한다.
      if (retry &&
          mounted &&
          !_inBackground &&
          _cam == null &&
          _error == null) {
        unawaited(_initCamera(retry: false));
      }
    }
  }

  Future<void> _disposeTracked(CameraController cam) {
    late final Future<void> done;
    done = () async {
      try {
        await cam.dispose();
      } catch (e) {
        debugPrint('camera dispose failed: $e');
      } finally {
        if (identical(_releasing, done)) _releasing = null;
      }
    }();
    _releasing = done;
    return done;
  }

  void _fail(String message) {
    if (mounted) setState(() => _error = message);
  }

  Future<void> _startStream() async {
    final cam = _cam;
    if (cam == null || cam.value.isStreamingImages) return;
    _guide?.reset(); // 이전 스트림의 결과가 새 촬영 점수로 남지 않게 한다
    await cam.startImageStream(_onImage);
  }

  void _onImage(CameraImage image) {
    final guide = _guide;
    final desc = _desc;
    if (guide == null || desc == null || _saving || _encoding) return;
    if (!guide.readyForFrame) return;
    _encoding = true;
    final data = frameDataFromCameraImage(image, frameRotationFor(desc));
    unawaited(() async {
      try {
        await guide.onFrame(data);
      } catch (_) {
        // 인코딩 실패는 다음 프레임에서 다시 시도한다.
      } finally {
        _encoding = false;
      }
    }());
  }

  Future<void> _releaseCamera() async {
    final cam = _cam;
    if (cam == null) return;
    _cam = null;
    if (mounted) setState(() {});
    await _disposeTracked(cam);
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState lifecycle) {
    if (lifecycle == AppLifecycleState.paused) {
      _inBackground = true;
      unawaited(_releaseCamera());
    } else if (lifecycle == AppLifecycleState.resumed) {
      _inBackground = false;
      if (_guide != null) unawaited(_initCamera());
    }
  }

  Future<void> _shoot() async {
    final cam = _cam;
    final state = _state;
    final style = state?.selectedStyle;
    final guide = _guide;
    if (cam == null || state == null || style == null || guide == null) return;
    if (_saving) return;
    setState(() => _saving = true);
    try {
      if (cam.value.isStreamingImages) await cam.stopImageStream();
      final file = await cam.takePicture();
      final bytes = await file.readAsBytes();
      final score = guide.lastResult?.score ?? 0;
      Shot? shot;
      if (state.token != null) {
        try {
          shot = await state.api.saveShot(
            photo: bytes,
            styleId: style.id,
            score: score,
            guideUsed: !_guideOff,
          );
        } catch (e) {
          // 폰 분석 중 서버가 없으면 기록 없이 결과만 보여 준다(401은 다시 로그인).
          if (!_local || (e is AxisApiException && e.status == 401)) rethrow;
        }
      }
      if (!mounted) return;
      if (shot == null) {
        ScaffoldMessenger.of(context)
          ..hideCurrentSnackBar()
          ..showSnackBar(const SnackBar(content: Text('서버에 연결하면 기록에 저장돼요')));
      }
      final result =
          shot ??
          Shot(
            id: 'local',
            styleId: style.id,
            score: score,
            guideUsed: !_guideOff,
            url: '',
            createdAt: DateTime.now().toUtc().toIso8601String(),
          );
      await Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (_) => ResultScreen(shot: result, photo: bytes),
        ),
      );
    } on CameraException {
      if (mounted) {
        ScaffoldMessenger.of(context)
          ..hideCurrentSnackBar()
          ..showSnackBar(
            const SnackBar(content: Text('사진을 찍지 못했어요. 다시 시도해 주세요')),
          );
      }
    } on AxisApiException catch (e) {
      if (e.status == 401 && mounted) {
        await state.logout();
        if (mounted) Navigator.of(context).popUntil((r) => r.isFirst);
        return;
      }
      if (mounted) showServerError(context);
    } catch (_) {
      if (mounted) showServerError(context);
    } finally {
      if (mounted) {
        setState(() => _saving = false);
        try {
          await _startStream();
        } catch (_) {
          _fail('카메라를 사용할 수 없어요');
        }
      }
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    SystemChrome.setPreferredOrientations(DeviceOrientation.values);
    _guide?.dispose();
    final cam = _cam;
    _cam = null;
    cam?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final error = _error;
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: Text(
          _state?.selectedStyle?.name ?? '촬영',
          style: const TextStyle(fontSize: 16),
          overflow: TextOverflow.ellipsis,
        ),
        actions: [
          if (error == null && !_local)
            IconButton(
              tooltip: '깊이 보기',
              icon: Icon(_depthOn ? Icons.layers : Icons.layers_outlined),
              color: _depthOn ? Colors.amberAccent : Colors.white,
              onPressed: () => setState(() => _depthOn = !_depthOn),
            ),
          Row(
            children: [
              const Text('가이드 끄기', style: TextStyle(color: Colors.white)),
              Switch(
                value: _guideOff,
                onChanged: (v) => setState(() => _guideOff = v),
              ),
            ],
          ),
        ],
      ),
      body: error != null
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      error,
                      textAlign: TextAlign.center,
                      style: const TextStyle(color: Colors.white),
                    ),
                    const SizedBox(height: 16),
                    FilledButton(
                      onPressed: () {
                        setState(() => _error = null);
                        if (_cam != null) {
                          _startStream().catchError(
                            (_) => _fail('카메라를 사용할 수 없어요'),
                          );
                        } else {
                          _initCamera();
                        }
                      },
                      child: const Text('다시 시도'),
                    ),
                  ],
                ),
              ),
            )
          : _buildCamera(),
    );
  }

  Widget _buildCamera() {
    final cam = _cam;
    final guide = _guide;
    if (cam == null || guide == null || !cam.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }
    return ListenableBuilder(
      listenable: guide,
      builder: (context, _) {
        final shown = (_guideOff || guide.disconnected) ? null : guide.shown;
        final ok = shown?.direction == 'OK';
        final last = guide.lastResult;
        final boxes =
            (!guide.disconnected &&
                last != null &&
                last.confidence >= minConfidence)
            ? last
            : null;
        // 안정화 전의 새 방향이면 목표 박스가 화살표와 어긋나므로 그리지 않는다.
        final showTarget = last?.direction == shown?.direction;
        return Column(
          children: [
            if (guide.disconnected)
              Container(
                width: double.infinity,
                color: Colors.red.shade700,
                padding: const EdgeInsets.all(8),
                child: const Text(
                  '서버에 연결할 수 없어요',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.white),
                ),
              ),
            Expanded(
              child: Center(
                child: AspectRatio(
                  aspectRatio: 1 / cam.value.aspectRatio,
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      CameraPreview(cam),
                      if (!_guideOff)
                        CustomPaint(
                          painter: GuideOverlayPainter(
                            boxes,
                            showTarget: showTarget,
                          ),
                        ),
                      GuideHud(
                        result: shown,
                        rotate: shown == null ? null : guide.shownRotate,
                      ),
                      if (last != null)
                        Positioned(
                          left: 8,
                          top: 56,
                          child: Text(
                            '안내: ${scorerLabel(last.scorer)}${_local ? ' · 폰' : ''} · ${last.timingMs['total'] ?? 0}ms',
                            style: const TextStyle(
                              color: Colors.white70,
                              fontSize: 12,
                              shadows: [
                                Shadow(blurRadius: 4, color: Colors.black),
                              ],
                            ),
                          ),
                        ),
                      if (_depthOn)
                        Positioned(
                          right: 8,
                          bottom: 72,
                          child: DepthPreview(base64Png: last?.depthPreview),
                        ),
                      if (ok)
                        IgnorePointer(
                          child: DecoratedBox(
                            decoration: BoxDecoration(
                              border: Border.all(
                                color: Colors.greenAccent,
                                width: 4,
                              ),
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 16),
              child: GestureDetector(
                onTap: _saving ? null : _shoot,
                child: Container(
                  width: 72,
                  height: 72,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: _saving
                        ? Colors.grey
                        : (ok ? Colors.greenAccent : Colors.white),
                    border: Border.all(
                      color: ok ? Colors.greenAccent : Colors.white54,
                      width: 4,
                    ),
                  ),
                  child: _saving
                      ? const Padding(
                          padding: EdgeInsets.all(22),
                          child: CircularProgressIndicator(strokeWidth: 3),
                        )
                      : null,
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}
