import 'package:flutter/material.dart';

import 'camera_screen.dart';
import 'history_screen.dart';

void openCamera(BuildContext context) =>
    Navigator.of(context)
        .push(MaterialPageRoute<void>(builder: (_) => const CameraScreen()));

void openHistory(BuildContext context) =>
    Navigator.of(context)
        .push(MaterialPageRoute<void>(builder: (_) => const HistoryScreen()));
