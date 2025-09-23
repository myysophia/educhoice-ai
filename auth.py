"""简单密码登录适配 Vanna Flask 的认证接口"""

from typing import Any, Dict, List

import flask
from vanna.flask.auth import AuthInterface


class SimplePassword(AuthInterface):
    """基于固定账号的简单密码认证"""

    def __init__(self, users: List[Dict[str, str]]):
        self.users = users

    def get_user(self, flask_request) -> Any:
        return flask_request.cookies.get("user")

    def is_logged_in(self, user: Any) -> bool:
        return user is not None

    def override_config_for_user(self, user: Any, config: Dict) -> Dict:
        return config

    def login_form(self) -> str:
        return """
  <div class="p-4 sm:p-7">
    <div class="text-center">
      <h1 class="block text-2xl font-bold text-gray-800 dark:text-white">登录问数智能体</h1>
      <p class="mt-2 text-sm text-gray-600 dark:text-gray-400">默认账号：admin / 密码：demo</p>
    </div>

    <div class="mt-5">
      <form action="/auth/login" method="POST">
        <div class="grid gap-y-4">
          <div>
            <label for="email" class="block text-sm mb-2 dark:text-white">账号</label>
            <div class="relative">
              <input type="text" id="email" name="email" class="py-3 px-4 block w-full border border-gray-200 rounded-lg text-sm focus:border-blue-500 focus:ring-blue-500 disabled:opacity-50 disabled:pointer-events-none dark:bg-slate-900 dark:border-gray-700 dark:text-gray-400 dark:focus:ring-gray-600" required>
            </div>
          </div>

          <div>
            <div class="flex justify-between items-center">
              <label for="password" class="block text-sm mb-2 dark:text-white">密码</label>
            </div>
            <div class="relative">
              <input type="password" id="password" name="password" class="py-3 px-4 block w-full border border-gray-200 rounded-lg text-sm focus:border-blue-500 focus:ring-blue-500 disabled:opacity-50 disabled:pointer-events-none dark:bg-slate-900 dark:border-gray-700 dark:text-gray-400 dark:focus:ring-gray-600" required>
            </div>
          </div>

          <button type="submit" class="w-full py-3 px-4 inline-flex justify-center items-center gap-x-2 text-sm font-semibold rounded-lg border border-transparent bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:pointer-events-none">立即登录</button>
        </div>
      </form>
    </div>
  </div>
        """

    def login_handler(self, flask_request) -> str:
        email = flask_request.form.get("email", "")
        password = flask_request.form.get("password", "")

        for user in self.users:
            if user.get("email") == email and user.get("password") == password:
                response = flask.make_response("Logged in as " + email)
                response.set_cookie("user", email)
                response.headers["Location"] = "/"
                response.status_code = 302
                return response

        return flask.make_response("Login failed", 401)

    def callback_handler(self, flask_request) -> str:
        user = flask_request.args.get("user", "")
        response = flask.make_response("Logged in as " + user)
        response.set_cookie("user", user)
        return response

    def logout_handler(self, flask_request) -> str:
        response = flask.make_response("Logged out")
        response.delete_cookie("user")
        return response
