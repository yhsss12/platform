/**
 * 登录页语言切换复核 — 在浏览器控制台粘贴运行。
 * 使用：打开 http://localhost:3001/login，F12 → Console，粘贴整段后回车。
 *
 * 1) 先执行「进入页面后」的读数；
 * 2) 再点击语言切换器选择「简体中文」，等待 300ms 后执行 readAfterSwitch()；
 * 3) 再选「English」，等待 300ms，再执行 readAfterSwitch()；
 * 4) 再选「Svenska」，等待 300ms，再执行 readAfterSwitch()。
 */

(function () {
  function readInitial() {
    var out = {
      'navigator.language': navigator.language,
      'localStorage.epi_locale': localStorage.getItem('epi_locale'),
    };
    console.log('=== 进入登录页后 ===');
    console.table(out);
    return out;
  }

  function readAfterSwitch() {
    var ls = localStorage.getItem('epi_locale');
    var form = document.querySelector('.auth-card form');
    if (!form) {
      console.warn('未找到 .auth-card form');
      return { epi_locale: ls, 登录按钮: null, 账号label: null, 协议: null };
    }
    var submitBtn = form.querySelector('button[type="submit"]');
    var labels = form.querySelectorAll('label');
    var agreement = form.querySelector('p');
    var btnText = submitBtn ? submitBtn.textContent.trim() : '';
    var labelText = labels.length ? labels[0].textContent.trim() : '';
    var agreementText = agreement ? agreement.textContent.trim() : '';
    var out = {
      epi_locale: ls,
      登录按钮文字: btnText,
      账号label首处: labelText,
      协议文字: agreementText,
    };
    console.log('=== 切换语言后（请确保已等待 200–500ms）===');
    console.table(out);
    return out;
  }

  window.__localeCheck = { readInitial: readInitial, readAfterSwitch: readAfterSwitch };
  console.log('已挂载：__localeCheck.readInitial() / __localeCheck.readAfterSwitch()');
  console.log('建议：先执行 readInitial()，再切换语言并等待 300ms 后执行 readAfterSwitch()。');
})();
