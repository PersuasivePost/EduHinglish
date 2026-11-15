import 'package:flutter/widgets.dart';
import 'package:google_sign_in_web/web_only.dart' as web;

/// Google's official "Sign in with Google" button; it reports back through GoogleSignIn.authenticationEvents.
Widget? buildWebGoogleButton() => web.renderButton();
