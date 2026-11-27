import 'package:flutter/material.dart';

const serverErrorText = '서버에 연결할 수 없어요. 설정에서 주소를 확인하세요';

void showServerError(BuildContext context) {
  ScaffoldMessenger.of(context)
    ..hideCurrentSnackBar()
    ..showSnackBar(const SnackBar(content: Text(serverErrorText)));
}
