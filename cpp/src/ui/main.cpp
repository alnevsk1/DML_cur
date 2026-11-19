#include "session.hpp"

#include <windows.h>
#include <windowsx.h>
#include <commctrl.h>
#include <commdlg.h>
#include <shellapi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace {
using namespace dml::ui;
using Clock = std::chrono::steady_clock;
constexpr wchar_t windowClass[] = L"DmlSam2VideoWindow";
constexpr COLORREF paper = RGB(229, 237, 243);
constexpr COLORREF ink = RGB(24, 45, 59);
constexpr COLORREF screen = RGB(13, 28, 38);
constexpr COLORREF screenText = RGB(190, 211, 222);
constexpr COLORREF accent = RGB(49, 199, 194);
constexpr COLORREF alert = RGB(151, 38, 41);

enum Control : int {
  openId = 100, playId, stepId, resetId, timelineId, precisionId,
  colorId, opacityId, outlineId, filenameId, readoutId, statusId, realtimeId
};

std::wstring wide(const std::string &text) {
  UINT codePage = CP_UTF8;
  int size = MultiByteToWideChar(codePage, MB_ERR_INVALID_CHARS, text.c_str(),
                                 -1, nullptr, 0);
  if (!size) {
    codePage = CP_ACP;
    size = MultiByteToWideChar(codePage, 0, text.c_str(), -1, nullptr, 0);
  }
  if (!size)
    return L"Неизвестная ошибка";
  std::wstring output(static_cast<size_t>(size), L'\0');
  MultiByteToWideChar(codePage, 0, text.c_str(), -1, output.data(), size);
  output.pop_back();
  return output;
}

std::wstring timecode(double seconds) {
  const auto ms = static_cast<long long>(seconds * 1000 + 0.5);
  std::wostringstream out;
  out << std::setfill(L'0') << std::setw(2) << ms / 60000 << L':'
      << std::setw(2) << ms / 1000 % 60 << L'.' << std::setw(3) << ms % 1000;
  return out.str();
}

class Window {
public:
  Window(HINSTANCE instance, std::filesystem::path root)
      : instance_(instance), session_(std::move(root)),
        dpi_(static_cast<int>(GetDpiForSystem())) {}

  ~Window() {
    if (paintBuffer_)
      DeleteDC(paintBuffer_);
    if (paintBitmap_)
      DeleteObject(paintBitmap_);
    for (HGDIOBJ object : std::array<HGDIOBJ, 5>{bodyFont_, headingFont_,
                           dataFont_, paperBrush_, screenBrush_})
      if (object)
        DeleteObject(object);
  }

  HWND create() {
    WNDCLASSW wc{};
    wc.lpfnWndProc = windowProc;
    wc.hInstance = instance_;
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hIcon = LoadIconW(nullptr, IDI_APPLICATION);
    wc.lpszClassName = windowClass;
    if (!RegisterClassW(&wc))
      throw std::runtime_error("Не удалось зарегистрировать класс окна.");
    HWND window = CreateWindowExW(
        WS_EX_ACCEPTFILES, windowClass, L"DML_cur  |  SAM 2.1-tiny",
        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN, CW_USEDEFAULT, CW_USEDEFAULT,
        px(1080), px(720), nullptr, nullptr, instance_, this);
    if (!window)
      throw std::runtime_error("Не удалось создать окно приложения.");
    return window;
  }

  void open(std::filesystem::path path) {
    frame_ = {};
    latestTrack_.reset();
    playing_ = false;
    dragging_ = false;
    Command command{Action::open};
    command.path = std::move(path);
    command.precision = precision_;
    submit(std::move(command));
    refreshImage();
  }

private:
  static LRESULT CALLBACK windowProc(HWND window, UINT message, WPARAM wParam,
                                     LPARAM lParam) {
    if (message == WM_NCCREATE) {
      auto *self = static_cast<Window *>(
          reinterpret_cast<CREATESTRUCTW *>(lParam)->lpCreateParams);
      self->hwnd_ = window;
      SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
    }
    auto *self =
        reinterpret_cast<Window *>(GetWindowLongPtrW(window, GWLP_USERDATA));
    if (self) {
      try {
        return self->handle(message, wParam, lParam);
      } catch (const std::exception &error) {
        if (message == WM_CREATE)
          return -1;
        self->playing_ = false;
        MessageBoxW(window, wide(error.what()).c_str(), L"DML_cur", MB_ICONERROR);
        return 0;
      }
    }
    return DefWindowProcW(window, message, wParam, lParam);
  }

  int px(int value) const { return MulDiv(value, dpi_, 96); }

  HFONT font(int size, int weight, const wchar_t *family) const {
    return CreateFontW(-px(size), 0, 0, 0, weight, FALSE, FALSE, FALSE,
                       DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                       CLEARTYPE_QUALITY, DEFAULT_PITCH, family);
  }

  HWND control(const wchar_t *type, const wchar_t *text, DWORD style,
               int id = 0, HFONT typeface = nullptr) {
    HWND child = CreateWindowExW(
        0, type, text, WS_CHILD | WS_VISIBLE | style, 0, 0, 0, 0, hwnd_,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)), instance_, nullptr);
    if (!child)
      throw std::runtime_error("Не удалось создать элемент управления.");
    SendMessageW(child, WM_SETFONT,
                 reinterpret_cast<WPARAM>(typeface ? typeface : bodyFont_), 0);
    return child;
  }

  void createControls() {
    paperBrush_ = CreateSolidBrush(paper);
    screenBrush_ = CreateSolidBrush(screen);
    bodyFont_ = font(15, FW_NORMAL, L"Segoe UI");
    headingFont_ = font(23, FW_SEMIBOLD, L"Segoe UI");
    dataFont_ = font(14, FW_NORMAL, L"Consolas");
    open_ = control(L"BUTTON", L"Открыть видео…", WS_TABSTOP, openId);
    filename_ = control(L"STATIC", L"", SS_PATHELLIPSIS, filenameId);
    timeline_ = control(TRACKBAR_CLASSW, L"Шкала времени", WS_TABSTOP | TBS_NOTICKS,
                        timelineId);
    play_ = control(L"BUTTON", L"Пуск", WS_TABSTOP, playId);
    step_ = control(L"BUTTON", L"Следующий кадр", WS_TABSTOP, stepId);
    reset_ = control(L"BUTTON", L"Сбросить трек", WS_TABSTOP, resetId);
    heading_ = control(L"STATIC", L"Настройки трекинга", 0, 0, headingFont_);
    precisionLabel_ = control(L"STATIC", L"Точность вычислений", 0);
    precisionBox_ = control(L"COMBOBOX", L"", WS_TABSTOP | CBS_DROPDOWNLIST,
                            precisionId);
    SendMessageW(precisionBox_, CB_ADDSTRING, 0,
                 reinterpret_cast<LPARAM>(L"FP32"));
    SendMessageW(precisionBox_, CB_ADDSTRING, 0,
                 reinterpret_cast<LPARAM>(L"Смешанная (декодер FP32)"));
    SendMessageW(precisionBox_, CB_SETCURSEL, 0, 0);
    colorButton_ = control(L"BUTTON", L"Цвет маски…", WS_TABSTOP, colorId);
    opacityLabel_ = control(L"STATIC", L"", 0);
    opacity_ = control(TRACKBAR_CLASSW, L"Непрозрачность маски", WS_TABSTOP | TBS_NOTICKS,
                       opacityId);
    SendMessageW(opacity_, TBM_SETRANGEMAX, FALSE, 100);
    SendMessageW(opacity_, TBM_SETPOS, TRUE, alpha_);
    outline_ = control(L"BUTTON", L"Показывать контур", WS_TABSTOP | BS_AUTOCHECKBOX,
                       outlineId);
    SendMessageW(outline_, BM_SETCHECK, BST_CHECKED, 0);
    realtimeBox_ = control(L"BUTTON", L"Режим реального времени",
                           WS_TABSTOP | BS_AUTOCHECKBOX, realtimeId);
    readout_ = control(L"STATIC", L"", 0, readoutId, dataFont_);
    status_ = control(L"STATIC", L"", 0, statusId);
    updateOpacity();
    updateControls();
    if (!SetTimer(hwnd_, 1, 15, nullptr))
      throw std::runtime_error("Не удалось запустить таймер воспроизведения.");
  }

  void layout() {
    RECT client{};
    GetClientRect(hwnd_, &client);
    const int width = client.right;
    const int height = client.bottom;
    const int margin = px(18);
    const int panel = px(240);
    const int sidebar = width - margin - panel;
    const int right = sidebar - px(18);
    const int bottom = height - px(86);
    canvas_ = {margin, px(58), right, bottom};
    swatch_ = {sidebar + px(132), px(173), sidebar + px(169), px(193)};
    auto move = [this](HWND child, int x, int y, int w, int h) {
      MoveWindow(child, x, y, std::max(w, 0), std::max(h, 0), TRUE);
    };
    move(open_, margin, px(13), px(136), px(31));
    move(filename_, margin + px(150), px(19), right - margin - px(150), px(24));
    move(timeline_, margin, bottom + px(5), right - margin, px(27));
    move(play_, margin, bottom + px(39), px(88), px(30));
    move(step_, margin + px(98), bottom + px(39), px(132), px(30));
    move(reset_, margin + px(240), bottom + px(39), px(116), px(30));
    move(heading_, sidebar, px(56), panel, px(32));
    move(precisionLabel_, sidebar, px(101), panel, px(21));
    move(precisionBox_, sidebar, px(125), panel, px(100));
    move(colorButton_, sidebar, px(167), px(120), px(31));
    move(opacityLabel_, sidebar, px(212), panel, px(20));
    move(opacity_, sidebar, px(233), panel, px(26));
    move(outline_, sidebar, px(268), panel, px(24));
    move(realtimeBox_, sidebar, px(302), panel, px(24));
    move(readout_, sidebar, px(347), panel, px(92));
    move(status_, sidebar, px(449), panel, height - px(465));
    refreshImage();
    InvalidateRect(hwnd_, nullptr, FALSE);
  }

  bool hasFrame() const { return !frame_.bgr.empty(); }

  bool navigating() const {
    return hasFrame() && (!busy_ || *busy_ == Action::advance);
  }

  bool canAdvance() const {
    return hasFrame() && !frame_.eof;
  }

  void updateOpacity() const {
    SetWindowTextW(opacityLabel_, (L"Непрозрачность маски  " + std::to_wstring(alpha_) +
                                   L"%").c_str());
  }

  void updateControls() {
    EnableWindow(timeline_, navigating() && frame_.count > 1);
    EnableWindow(play_, playing_ || (navigating() && canAdvance()));
    EnableWindow(step_, !playing_ && !busy_ && !dragging_ && canAdvance());
    EnableWindow(reset_, navigating());
    EnableWindow(precisionBox_, !busy_ || *busy_ == Action::advance);
    EnableWindow(realtimeBox_, !playing_ && !busy_ && !dragging_);
    SetWindowTextW(play_, playing_ ? L"Пауза" : L"Пуск");

    if (!dragging_) {
      SendMessageW(timeline_, TBM_SETRANGEMAX, FALSE,
                   frame_.count > 0 ? frame_.count - 1 : 0);
      SendMessageW(timeline_, TBM_SETPOS, TRUE, frame_.index);
    }
    SetWindowTextW(filename_, hasFrame() ? frame_.path.filename().c_str()
                                        : L"Видео не открыто");
    std::wostringstream info;
    if (hasFrame()) {
      info << L"Кадр " << frame_.index + 1 << L" / " << frame_.count << L"\r\n"
           << timecode(frame_.index / frame_.fps) << L" / "
           << timecode(frame_.count / frame_.fps) << L"\r\n";
      if (latestTrack_) {
        info << std::fixed << std::setprecision(1) << latestTrack_->latencyMs
             << L" мс на обработку\r\n";
        if (realtime_)
          info << L"Кадр трека " << latestTrack_->index + 1 << L"\r\nПропущено "
               << latestTrack_->skipped;
      } else {
        info << L"Обработка \u2014";
      }
    } else {
      info << L"Кадр \u2014\r\nДлительность \u2014\r\nОбработка \u2014";
    }
    SetWindowTextW(readout_, info.str().c_str());

    const wchar_t *status = L"Откройте видео или перетащите файл в это окно.";
    if (busy_) {
      switch (*busy_) {
      case Action::open: status = L"Открываем видео…"; break;
      case Action::select: status = L"Выбираем объект…"; break;
      case Action::seek: status = L"Переходим к кадру…"; break;
      case Action::reset: status = L"Сбрасываем трек…"; break;
      case Action::precision: status = L"Меняем точность вычислений…"; break;
      case Action::advance: break;
      }
    }
    if ((!busy_ || *busy_ == Action::advance) && hasFrame()) {
      if (dragging_)
        status = L"Отпустите ползунок, чтобы перейти к кадру и сбросить трек.";
      else if (playing_ && frame_.tracking)
        status = realtime_ ? L"Перетащите превью за заголовок. Размер меняется за угол."
                           : L"Отслеживаем выбранный объект.";
      else if (playing_)
        status = L"Поставьте видео на паузу, чтобы выбрать объект.";
      else if (!canAdvance())
        status = L"Видео закончилось. Выберите другой кадр на шкале времени.";
      else if (!frame_.tracking)
        status = L"Щёлкните по объекту на видео, чтобы начать трекинг.";
      else
        status = L"Объект выбран. Нажмите «Пуск» или «Следующий кадр».";
    }
    auto text = frame_.error.empty() ? std::wstring(status) : wide(frame_.error);
    if (text != statusText_) {
      SetWindowTextW(status_, text.c_str());
      statusText_ = std::move(text);
    }
  }

  void submit(Command command) {
    frame_.error.clear();
    busy_ = command.action;
    advanceRealtime_ = command.realtime;
    session_.submit(std::move(command));
    updateControls();
  }

  void clearTrackAt(Action action, int index) {
    playing_ = false;
    dragging_ = false;
    frame_.tracking = false;
    frame_.seed.reset();
    frame_.mask.release();
    frame_.latencyMs = 0;
    latestTrack_.reset();
    Command command{action};
    command.index = index;
    command.precision = precision_;
    submit(std::move(command));
    refreshImage();
  }

  void tick() {
    bool changed = false;
    if (auto result = session_.takeResult()) {
      if ((advanceRealtime_ || result->eof) && result->error.empty()) {
        result->tracking = frame_.tracking;
        result->error = frame_.error;
      }
      frame_ = std::move(*result);
      busy_.reset();
      if (!frame_.error.empty() || !canAdvance())
        playing_ = false;
      if (!frame_.tracking)
        latestTrack_.reset();
      else if (!frame_.mask.empty())
        latestTrack_ = frame_;
      changed = true;
    }
    if (auto tracked = session_.takeTrackedResult(); tracked && frame_.tracking) {
      if (!tracked->error.empty()) {
        frame_.tracking = false;
        frame_.mask.release();
        frame_.seed.reset();
        latestTrack_.reset();
        playing_ = false;
        frame_.error = std::move(tracked->error);
        changed = true;
      } else if (!latestTrack_ || tracked->index > latestTrack_->index) {
        latestTrack_ = std::move(*tracked);
        changed = true;
      }
    }
    if (changed) {
      updateControls();
      refreshImage();
    }
    if (playing_ && !busy_ && !dragging_ && canAdvance() && playbackDue())
      advance(realtime_);
  }

  bool playbackDue() const {
    const auto now = Clock::now();
    if (realtime_) {
      const auto next = frame_.index + 1 - startFrame_;
      return now - playbackStart_ >= std::chrono::duration<double>(next / frame_.fps);
    }
    return now - lastAdvance_ >= std::chrono::duration<double>(1 / frame_.fps);
  }

  void advance(bool realtime = false) {
    lastAdvance_ = Clock::now();
    Command command{Action::advance};
    command.track = frame_.tracking;
    command.realtime = realtime;
    submit(std::move(command));
  }

  void togglePlay() {
    if (playing_) {
      playing_ = false;
      updateControls();
    } else if (navigating() && canAdvance() && !dragging_) {
      playing_ = true;
      lastAdvance_ = Clock::time_point{};
      playbackStart_ = Clock::now();
      startFrame_ = frame_.index;
      updateControls();
    }
  }

  void chooseVideo() {
    playing_ = false;
    updateControls();
    std::vector<wchar_t> path(32768);
    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = hwnd_;
    dialog.lpstrTitle = L"Открыть видео";
    dialog.lpstrFilter =
        L"Видеофайлы\0*.mp4;*.avi;*.mov;*.mkv;*.webm;*.m4v;*.wmv\0Все файлы\0*.*\0";
    dialog.lpstrFile = path.data();
    dialog.nMaxFile = static_cast<DWORD>(path.size());
    dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (GetOpenFileNameW(&dialog))
      open(std::filesystem::path(path.data()));
    else if (CommDlgExtendedError())
      showError("Не удалось открыть окно выбора видео.");
  }

  void dropVideo(HDROP drop) {
    std::unique_ptr<std::remove_pointer_t<HDROP>, decltype(&DragFinish)>
        handle(drop, DragFinish);
    if (DragQueryFileW(drop, 0xFFFFFFFF, nullptr, 0) != 1) {
      showError("Перетащите один видеофайл.");
      return;
    }
    std::wstring path(DragQueryFileW(drop, 0, nullptr, 0) + 1, L'\0');
    const UINT length = DragQueryFileW(drop, 0, path.data(),
                                       static_cast<UINT>(path.size()));
    if (length) {
      path.resize(length);
      open(std::filesystem::path(path));
    }
  }

  void showError(const std::string &error) {
    playing_ = false;
    frame_.error = error;
    updateControls();
  }

  void chooseColor() {
    CHOOSECOLORW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = hwnd_;
    dialog.rgbResult = maskColor_;
    dialog.lpCustColors = customColors_.data();
    dialog.Flags = CC_FULLOPEN | CC_RGBINIT;
    if (ChooseColorW(&dialog)) {
      maskColor_ = dialog.rgbResult;
      InvalidateRect(hwnd_, &swatch_, FALSE);
      refreshImage();
    }
  }

  void onCommand(int id, int notification) {
    switch (id) {
    case openId: chooseVideo(); break;
    case playId: togglePlay(); break;
    case stepId:
      if (!playing_ && !busy_ && !dragging_ && canAdvance())
        advance();
      break;
    case resetId:
      if (navigating())
        clearTrackAt(Action::reset, frame_.index);
      break;
    case precisionId:
      if (notification == CBN_SELCHANGE) {
        const auto mode = SendMessageW(precisionBox_, CB_GETCURSEL, 0, 0) == 1
                              ? Precision::mixed : Precision::fp32;
        if (mode != precision_) {
          precision_ = mode;
          if (navigating())
            clearTrackAt(Action::precision, frame_.index);
        }
      }
      break;
    case realtimeId:
      if (!playing_ && !busy_ && !dragging_) {
        realtime_ = SendMessageW(realtimeBox_, BM_GETCHECK, 0, 0) == BST_CHECKED;
        updateControls();
        refreshImage();
      }
      break;
    case colorId: chooseColor(); break;
    case outlineId:
      drawOutline_ = SendMessageW(outline_, BM_GETCHECK, 0, 0) == BST_CHECKED;
      refreshImage();
      break;
    }
  }

  void onScroll(HWND controlWindow, int code) {
    if (controlWindow == opacity_) {
      alpha_ = static_cast<int>(SendMessageW(opacity_, TBM_GETPOS, 0, 0));
      updateOpacity();
      refreshImage();
    } else if (controlWindow == timeline_ && navigating()) {
      if (code == TB_THUMBTRACK) {
        if (!dragging_) {
          playing_ = false;
          dragging_ = true;
          updateControls();
        }
      } else if (code != TB_ENDTRACK || dragging_) {
        const int index =
            static_cast<int>(SendMessageW(timeline_, TBM_GETPOS, 0, 0));
        if (index != frame_.index || dragging_)
          clearTrackAt(Action::seek, index);
      }
    }
  }

  void select(POINT p) {
    SetFocus(hwnd_);
    if (busy_ || playing_ || dragging_ || frame_.tracking || !hasFrame() ||
        !PtInRect(&image_, p))
      return;
    Command command{Action::select};
    command.index = frame_.index;
    command.precision = precision_;
    command.point = {
        static_cast<float>((p.x - image_.left) * (frame_.bgr.cols * 1.0) /
                           (image_.right - image_.left)),
        static_cast<float>((p.y - image_.top) * (frame_.bgr.rows * 1.0) /
                           (image_.bottom - image_.top))};
    submit(std::move(command));
  }

  cv::Mat compose(const cv::Mat &bgr, const cv::Mat &originalMask,
                  cv::Size size) const {
    cv::Mat display;
    cv::resize(bgr, display, size, 0, 0,
               size.width < bgr.cols ? cv::INTER_AREA : cv::INTER_LINEAR);
    if (!originalMask.empty() && (alpha_ || drawOutline_)) {
      cv::Mat mask;
      cv::resize(originalMask, mask, size, 0, 0, cv::INTER_NEAREST);
      const cv::Scalar color(GetBValue(maskColor_), GetGValue(maskColor_),
                             GetRValue(maskColor_));
      if (alpha_) {
        cv::Mat tint(size, CV_8UC3, color), blended;
        cv::addWeighted(display, 1 - alpha_ / 100.0, tint, alpha_ / 100.0,
                        0, blended);
        blended.copyTo(display, mask);
      }
      if (drawOutline_) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(display, contours, -1, color, px(1));
      }
    }
    cv::Mat bgra;
    cv::cvtColor(display, bgra, cv::COLOR_BGR2BGRA);
    return bgra;
  }

  void refreshImage() {
    dib_.release();
    image_ = {};
    const int w = canvas_.right - canvas_.left;
    const int h = canvas_.bottom - canvas_.top;
    if (hasFrame() && w > 0 && h > 0) {
      const double scale =
          std::min(w / static_cast<double>(frame_.bgr.cols),
                   h / static_cast<double>(frame_.bgr.rows));
      const cv::Size size(std::max(1, static_cast<int>(frame_.bgr.cols * scale)),
                          std::max(1, static_cast<int>(frame_.bgr.rows * scale)));
      const int x = canvas_.left + (w - size.width) / 2;
      const int y = canvas_.top + (h - size.height) / 2;
      image_ = {x, y, x + size.width, y + size.height};

      const bool sameFrame = latestTrack_ && latestTrack_->index == frame_.index;
      dib_ = compose(frame_.bgr, sameFrame ? latestTrack_->mask : frame_.mask, size);
    }
    refreshInset();
    InvalidateRect(hwnd_, &canvas_, FALSE);
  }

  std::pair<int, int> insetWidths() const {
    const int w = image_.right - image_.left;
    const int h = image_.bottom - image_.top;
    const auto &tracked = latestTrack_->bgr;
    const auto fitWidth = [&](int width, int height) {
      return std::max(1, std::min(width, static_cast<int>(
          height * (tracked.cols / static_cast<double>(tracked.rows)))));
    };
    return {fitWidth(std::min(w / 3, px(250)), h / 3),
            fitWidth(w - px(16), h - px(36))};
  }

  void refreshInset() {
    insetDib_.release();
    inset_ = {};
    if (!realtime_ || !latestTrack_ || latestTrack_->index >= frame_.index ||
        latestTrack_->bgr.empty() || image_.right - image_.left < px(120) ||
        image_.bottom - image_.top < px(90))
      return;
    const auto [normal, maximum] = insetWidths();
    const int width = insetWidth_ ? std::clamp(insetWidth_, 1, maximum) : normal;
    const auto &tracked = latestTrack_->bgr;
    const int height = std::max(1, static_cast<int>(
        width * (tracked.rows / static_cast<double>(tracked.cols))));
    insetDib_ = compose(tracked, latestTrack_->mask, {width, height});
    placeInset();
  }

  void placeInset() {
    const long right = std::clamp(image_.right - px(8) - insetOffset_.x,
        image_.left + px(8) + insetDib_.cols, image_.right - px(8));
    const long bottom = std::clamp(image_.bottom - px(8) - insetOffset_.y,
        image_.top + px(28) + insetDib_.rows, image_.bottom - px(8));
    inset_ = {right - insetDib_.cols, bottom - insetDib_.rows, right, bottom};
  }

  RECT insetCard() const {
    return {inset_.left - px(2), inset_.top - px(20),
            inset_.right + px(2), inset_.bottom + px(2)};
  }

  RECT insetTitle() const {
    auto title = insetCard();
    title.bottom = inset_.top;
    return title;
  }

  RECT insetGrip() const {
    auto grip = insetTitle();
    grip.right = std::min(grip.right, grip.left + px(20));
    return grip;
  }

  void pointerDown(POINT p) {
    const auto card = insetCard();
    if (!insetDib_.empty() && PtInRect(&card, p)) {
      const auto title = insetTitle();
      if (PtInRect(&title, p)) {
        const auto grip = insetGrip();
        insetDrag_ = InsetDrag{p, inset_, PtInRect(&grip, p) != FALSE};
        SetCapture(hwnd_);
      }
      return;
    }
    select(p);
  }

  void dragInset(POINT p) {
    if (!insetDrag_ || insetDib_.empty())
      return;
    const auto &drag = *insetDrag_;
    const long dx = p.x - drag.start.x;
    const long dy = p.y - drag.start.y;
    const auto oldCard = insetCard();
    if (drag.resizing) {
      const auto [normal, maximum] = insetWidths();
      const auto &tracked = latestTrack_->bgr;
      const double aspect = tracked.rows / static_cast<double>(tracked.cols);
      // Project the pointer movement onto the aspect-ratio diagonal.
      const int delta = static_cast<int>(std::lround(
          (dx + dy * aspect) / (1 + aspect * aspect)));
      const int original = drag.rect.right - drag.rect.left;
      const int width = std::clamp(original - delta,
                                   std::min(normal, px(96)), maximum);
      if (width == inset_.right - inset_.left)
        return;
      insetWidth_ = width;
      refreshInset();
    } else {
      insetOffset_ = {
          std::clamp(image_.right - px(8) - drag.rect.right - dx, 0L,
                     image_.right - image_.left - px(16) - insetDib_.cols),
          std::clamp(image_.bottom - px(8) - drag.rect.bottom - dy, 0L,
                     image_.bottom - image_.top - px(36) - insetDib_.rows)};
      placeInset();
    }
    const auto newCard = insetCard();
    RECT dirty{};
    UnionRect(&dirty, &oldCard, &newCard);
    InvalidateRect(hwnd_, &dirty, FALSE);
  }

  static void drawDib(HDC dc, const RECT &rect, const cv::Mat &dib) {
    BITMAPINFO bitmap{};
    bitmap.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bitmap.bmiHeader.biWidth = dib.cols;
    bitmap.bmiHeader.biHeight = -dib.rows;
    bitmap.bmiHeader.biPlanes = 1;
    bitmap.bmiHeader.biBitCount = 32;
    bitmap.bmiHeader.biCompression = BI_RGB;
    SetDIBitsToDevice(dc, rect.left, rect.top, dib.cols, dib.rows,
                      0, 0, 0, dib.rows, dib.data, &bitmap, DIB_RGB_COLORS);
  }

  void paintInset(HDC dc) const {
    const auto card = insetCard();
    FillRect(dc, &card, screenBrush_);
    drawDib(dc, inset_, insetDib_);
    const auto grip = insetGrip();
    const auto oldPen = SelectObject(dc, GetStockObject(DC_PEN));
    const auto oldPenColor = SetDCPenColor(dc, screenText);
    for (const int offset : {8, 12, 16}) {
      MoveToEx(dc, grip.left + px(4), grip.top + px(offset), nullptr);
      LineTo(dc, grip.left + px(offset), grip.top + px(4));
    }
    SetDCPenColor(dc, oldPenColor);
    SelectObject(dc, oldPen);
    RECT label{grip.right + px(2), card.top, inset_.right, inset_.top};
    const auto title = L"Кадр " + std::to_wstring(latestTrack_->index + 1);
    const auto oldFont = SelectObject(dc, dataFont_);
    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, screenText);
    DrawTextW(dc, title.c_str(), -1, &label,
              DT_SINGLELINE | DT_VCENTER | DT_END_ELLIPSIS);
    SelectObject(dc, oldFont);
  }

  bool preparePaintBuffer(HDC dc, SIZE size) {
    if (size.cx <= 0 || size.cy <= 0)
      return false;
    if (paintBitmap_ && paintSize_.cx == size.cx && paintSize_.cy == size.cy)
      return true;
    if (!paintBuffer_)
      paintBuffer_ = CreateCompatibleDC(dc);
    if (!paintBuffer_)
      return false;
    const auto bitmap = CreateCompatibleBitmap(dc, size.cx, size.cy);
    if (!bitmap)
      return false;
    if (!SelectObject(paintBuffer_, bitmap)) {
      DeleteObject(bitmap);
      return false;
    }
    if (paintBitmap_)
      DeleteObject(paintBitmap_);
    paintBitmap_ = bitmap;
    paintSize_ = size;
    return true;
  }

  void paint() {
    PAINTSTRUCT ps{};
    HDC dc = BeginPaint(hwnd_, &ps);
    RECT client{};
    GetClientRect(hwnd_, &client);
    const auto &dirty = ps.rcPaint;
    if (preparePaintBuffer(dc, {client.right, client.bottom})) {
      drawWindow(paintBuffer_, dirty);
      BitBlt(dc, dirty.left, dirty.top, dirty.right - dirty.left,
             dirty.bottom - dirty.top, paintBuffer_, dirty.left, dirty.top,
             SRCCOPY);
    } else {
      drawWindow(dc, dirty);
    }
    EndPaint(hwnd_, &ps);
  }

  void drawWindow(HDC dc, const RECT &dirty) const {
    FillRect(dc, &dirty, paperBrush_);
    FillRect(dc, &canvas_, screenBrush_);
    if (!dib_.empty()) {
      drawDib(dc, image_, dib_);
      if (frame_.seed)
        paintSeed(dc);
      if (!insetDib_.empty())
        paintInset(dc);
    } else {
      SetBkMode(dc, TRANSPARENT);
      SetTextColor(dc, screenText);
      auto font = SelectObject(dc, headingFont_);
      RECT rect = canvas_;
      DrawTextW(dc, L"Предпросмотр видео", -1, &rect,
                DT_CENTER | DT_VCENTER | DT_SINGLELINE);
      SelectObject(dc, font);
    }
    HBRUSH swatch = CreateSolidBrush(maskColor_);
    FillRect(dc, &swatch_, swatch);
    DeleteObject(swatch);
    FrameRect(dc, &swatch_, static_cast<HBRUSH>(GetStockObject(DKGRAY_BRUSH)));
  }

  void paintSeed(HDC dc) const {
    const int x = image_.left +
        static_cast<int>(frame_.seed->x * dib_.cols / frame_.bgr.cols);
    const int y = image_.top +
        static_cast<int>(frame_.seed->y * dib_.rows / frame_.bgr.rows);
    const int outer = px(10), inner = px(4);
    const int saved = SaveDC(dc);
    IntersectClipRect(dc, image_.left, image_.top, image_.right, image_.bottom);
    HPEN pen = CreatePen(PS_SOLID, px(2), RGB(255, 255, 255));
    SelectObject(dc, pen);
    MoveToEx(dc, x - outer, y, nullptr);
    LineTo(dc, x - inner, y);
    MoveToEx(dc, x + inner, y, nullptr);
    LineTo(dc, x + outer, y);
    MoveToEx(dc, x, y - outer, nullptr);
    LineTo(dc, x, y - inner);
    MoveToEx(dc, x, y + inner, nullptr);
    LineTo(dc, x, y + outer);
    RestoreDC(dc, saved);
    DeleteObject(pen);
  }

  LRESULT handle(UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_CREATE: createControls(); return 0;
    case WM_SIZE:
      if (wParam != SIZE_MINIMIZED)
        layout();
      return 0;
    case WM_GETMINMAXINFO: {
      auto *info = reinterpret_cast<MINMAXINFO *>(lParam);
      info->ptMinTrackSize = {px(710), px(595)};
      return 0;
    }
    case WM_ERASEBKGND: return 1;
    case WM_PAINT: paint(); return 0;
    case WM_CTLCOLORSTATIC:
    case WM_CTLCOLORBTN: {
      auto dc = reinterpret_cast<HDC>(wParam);
      const bool error = reinterpret_cast<HWND>(lParam) == status_ &&
                         !frame_.error.empty();
      SetTextColor(dc, error ? alert : ink);
      SetBkColor(dc, paper);
      return reinterpret_cast<LRESULT>(paperBrush_);
    }
    case WM_COMMAND: onCommand(LOWORD(wParam), HIWORD(wParam)); return 0;
    case WM_HSCROLL:
      onScroll(reinterpret_cast<HWND>(lParam), LOWORD(wParam));
      return 0;
    case WM_TIMER: tick(); return 0;
    case WM_SETCURSOR:
      if (LOWORD(lParam) == HTCLIENT && !insetDib_.empty()) {
        POINT p{};
        const auto title = insetTitle();
        if (insetDrag_ || (GetCursorPos(&p) && ScreenToClient(hwnd_, &p) &&
                           PtInRect(&title, p))) {
          const auto grip = insetGrip();
          const bool resizing = insetDrag_ ? insetDrag_->resizing
                                          : PtInRect(&grip, p) != FALSE;
          SetCursor(LoadCursorW(nullptr, resizing ? IDC_SIZENWSE : IDC_SIZEALL));
          return TRUE;
        }
      }
      break;
    case WM_LBUTTONDOWN:
      pointerDown({GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)});
      return 0;
    case WM_MOUSEMOVE:
      dragInset({GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)});
      return 0;
    case WM_LBUTTONUP:
    case WM_CANCELMODE:
      if (insetDrag_ && GetCapture() == hwnd_)
        ReleaseCapture();
      insetDrag_.reset();
      break;
    case WM_CAPTURECHANGED:
      insetDrag_.reset();
      return 0;
    case WM_DROPFILES: dropVideo(reinterpret_cast<HDROP>(wParam)); return 0;
    case WM_KEYDOWN:
      if (!(lParam & (1L << 30)) && wParam == VK_SPACE)
        togglePlay();
      return 0;
    case WM_DESTROY:
      KillTimer(hwnd_, 1);
      PostQuitMessage(0);
      return 0;
    }
    return DefWindowProcW(hwnd_, message, wParam, lParam);
  }

  HINSTANCE instance_;
  Session session_;
  int dpi_;
  HWND hwnd_ = nullptr;
  HWND open_ = nullptr, filename_ = nullptr, timeline_ = nullptr, play_ = nullptr,
       step_ = nullptr, reset_ = nullptr, heading_ = nullptr,
       precisionLabel_ = nullptr, precisionBox_ = nullptr, colorButton_ = nullptr,
       opacityLabel_ = nullptr, opacity_ = nullptr, outline_ = nullptr,
       readout_ = nullptr, status_ = nullptr, realtimeBox_ = nullptr;
  HFONT bodyFont_ = nullptr, headingFont_ = nullptr, dataFont_ = nullptr;
  HBRUSH paperBrush_ = nullptr, screenBrush_ = nullptr;
  HDC paintBuffer_ = nullptr;
  HBITMAP paintBitmap_ = nullptr;
  SIZE paintSize_{};
  std::wstring statusText_;
  RECT canvas_{}, image_{}, swatch_{}, inset_{};
  struct InsetDrag {
    POINT start;
    RECT rect;
    bool resizing;
  };
  std::optional<InsetDrag> insetDrag_;
  POINT insetOffset_{};
  int insetWidth_ = 0;
  cv::Mat dib_, insetDib_;
  Frame frame_;
  std::optional<Frame> latestTrack_;
  std::optional<Action> busy_;
  Clock::time_point lastAdvance_{}, playbackStart_{};
  int startFrame_ = 0;
  Precision precision_ = Precision::fp32;
  COLORREF maskColor_ = accent;
  std::array<COLORREF, 16> customColors_{};
  int alpha_ = 40;
  bool drawOutline_ = true, playing_ = false, dragging_ = false;
  bool realtime_ = false, advanceRealtime_ = false;
};
} // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int show) try {
  SetProcessDPIAware();
  INITCOMMONCONTROLSEX controls{sizeof(controls), ICC_BAR_CLASSES};
  if (!InitCommonControlsEx(&controls))
    throw std::runtime_error("Не удалось инициализировать элементы управления Windows.");
  int argc = 0;
  std::unique_ptr<LPWSTR, decltype(&LocalFree)> argv(
      CommandLineToArgvW(GetCommandLineW(), &argc), LocalFree);
  if (!argv || argc > 2)
    throw std::runtime_error("В аргументах можно указать только один видеофайл.");
  Window app(instance, std::filesystem::current_path());
  HWND window = app.create();
  ShowWindow(window, show);
  if (argc == 2)
    app.open(std::filesystem::path(argv.get()[1]));

  ACCEL shortcuts[] = {
      {FVIRTKEY | FCONTROL, 'O', openId},
      {FVIRTKEY | FCONTROL, 'R', resetId},
  };
  HACCEL keys = CreateAcceleratorTableW(shortcuts, 2);
  MSG msg{};
  BOOL next = FALSE;
  while ((next = GetMessageW(&msg, nullptr, 0, 0)) > 0) {
    if ((!keys || !TranslateAcceleratorW(window, keys, &msg)) &&
        !IsDialogMessageW(window, &msg)) {
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }
  }
  if (keys)
    DestroyAcceleratorTable(keys);
  return next < 0 ? 1 : static_cast<int>(msg.wParam);
} catch (const std::exception &error) {
  MessageBoxW(nullptr, wide(error.what()).c_str(), L"DML_cur", MB_ICONERROR);
  return 1;
}
