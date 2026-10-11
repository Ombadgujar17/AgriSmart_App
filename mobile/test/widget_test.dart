import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:agrismart_app/main.dart';

void main() {
  testWidgets('AgriSmartApp smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const AgriSmartApp());
    expect(find.byType(MaterialApp), findsOneWidget);
  });
}
