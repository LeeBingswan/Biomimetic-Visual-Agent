import os
import warnings
import time
import random
import numpy as np
import mss
import uiautomation as auto
import pyperclip
import pyautogui
import keyboard
import cv2
import csv
import datetime
from collections import deque
from rapidocr_onnxruntime import RapidOCR
from openai import OpenAI

warnings.filterwarnings("ignore")  # Suppress warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================= ⚙️ Flagship Configuration Area =================
API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-KEY")
TARGET_NAME = "target name"
MODEL_NAME = "deepseek-chat"
AUTO_CHAT_INTERVAL = 60
DEBUG_MODE = True
MAX_MEMORY_ROUNDS = 20
READING_SPEED = 0.05
TYPING_SPEED = 0.10
BURST_WAIT_TIME = 1.5
# =================================================

print(f"🚀 正在加载 RapidOCR...")
ocr = RapidOCR()

if "sk-" not in API_KEY:
    print("❌ 错误：请先配置正确的 DeepSeek API Key！")
    exit()

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com", timeout=20.0)


# ================= 📊 Academic Experiment Logger =================
class ExperimentLogger:
    def __init__(self, filename="paper_experiment_data.csv"):
        self.filename = filename
        if not os.path.exists(self.filename):  # Initialize header if file does not exist
            with open(self.filename, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp",
                    "Event_Type",
                    "T_capture_ms",
                    "T_ocr_ms",
                    "T_llm_ms",
                    "T_exec_ms",
                    "Simulated_Delay_s",
                    "Status"
                ])
            print(f"📊 已创建学术实验数据记录表: {self.filename}")

    def log_interaction(self, event_type, metrics, status="Success"):
        """Write the time cost of a complete interaction link into CSV"""
        try:
            with open(self.filename, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    event_type,
                    f"{metrics.get('t_capture', 0):.2f}",
                    f"{metrics.get('t_ocr', 0):.2f}",
                    f"{metrics.get('t_llm', 0):.2f}",
                    f"{metrics.get('t_exec', 0):.2f}",
                    f"{metrics.get('simulated_delay', 0):.3f}",
                    status
                ])
        except Exception as e:
            if DEBUG_MODE: print(f"⚠️ 日志写入失败: {e}")


logger = ExperimentLogger()


# =========================================================

class HumanBehavior:
    @staticmethod
    def simulate_reading(text):
        if not text: return 0
        delay = 0.5 + (len(text) * READING_SPEED) + random.uniform(0, 0.5)
        print(f"   👀 正在阅读 ({len(text)}字)... 耗时 {delay:.2f}s")
        time.sleep(delay)
        return delay

    @staticmethod
    def simulate_typing_delay(text):
        return 0.6 + (len(text) * TYPING_SPEED) + random.uniform(0, 0.5)

    @staticmethod
    def random_mouse_jitter():
        x, y = pyautogui.position()
        pyautogui.moveTo(x + random.randint(-5, 5), y + random.randint(-5, 5), duration=0.2)


class VisualWeChat:
    def __init__(self):
        self.sct = mss.mss()
        monitor = self.sct.monitors[1]
        self.screen_w = monitor["width"]
        self.screen_h = monitor["height"]
        self.memory = deque(maxlen=MAX_MEMORY_ROUNDS)
        self.last_processed_text = ""
        self.last_interaction_time = time.time()
        self.wechat_win = auto.WindowControl(searchDepth=1, Name="微信")
        self.latest_vision_metrics = {"t_capture": 0, "t_ocr": 0}  # Temporarily store time cost data of the perception layer

    def focus_target(self):
        if self.wechat_win.Exists(maxSearchSeconds=2):
            self.wechat_win.SetActive()
            time.sleep(0.5)
        else:
            print("⚠️ 未检测到微信窗口，请手动打开并置前")
            return False

        pyautogui.hotkey('ctrl', 'f')
        time.sleep(0.5)
        pyperclip.copy(TARGET_NAME)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(0.8)
        pyautogui.press('enter')
        time.sleep(0.5)
        return True

    def get_visible_incoming_messages(self):
        """Pure visual OCR extraction, perfectly immune to extremely small windows and DPI scaling"""
        try:
            t_start_capture = time.perf_counter()

            if not self.wechat_win.Exists(0.1):
                return []

            rect = self.wechat_win.BoundingRectangle
            win_width = rect.right - rect.left
            win_height = rect.bottom - rect.top

            if win_width < 400 or win_height < 300:
                return []

            # ================= 🚨 Core Fix 1: Dynamic Boundary and Right-side Safe Sampling =================
            left_offset = min(int(win_width * 0.45), 450)  # Use non-linear boundary: cut off up to 45% for small windows, up to 450 pixels for large windows (perfectly compatible with 150% DPI)
            left_start = rect.left + left_offset
            capture_width = win_width - left_offset

            top_start = rect.top + 60
            capture_height = win_height - 60 - 160

            if left_start < 0:
                capture_width += left_start
                left_start = 0
            if top_start < 0:
                capture_height += top_start
                top_start = 0

            monitor = {
                "top": int(max(0, top_start)),
                "left": int(max(0, left_start)),
                "width": int(max(50, capture_width)),
                "height": int(max(50, capture_height))
            }

            img_bgra = np.array(self.sct.grab(monitor))
            img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)

            sample_start = int(capture_width * 0.7)  # [Critical modification]: Absolutely cannot sample on the far left (0:10)! We sample on the right side (70%~80%) of the captured image, where it is 100% pure chat background and will never touch the underlying list
            bg_sample = img_bgr[:, sample_start:sample_start + 20]
            # =======================================================================

            bg_gray = cv2.cvtColor(bg_sample, cv2.COLOR_BGR2GRAY)
            bg_brightness = np.mean(bg_gray)

            is_light_mode = bg_brightness > 127

            if is_light_mode:
                lower_color = np.array([240, 240, 240])
                upper_color = np.array([255, 255, 255])
            else:
                lower_color = np.array([35, 35, 35])
                upper_color = np.array([65, 65, 65])

            mask = cv2.inRange(img_bgr, lower_color, upper_color)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            bubble_crops = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if 500 < area < 80000 and x < capture_width * 0.8:  # Relax x limit to prevent bubbles from being mistakenly killed in squeezed state
                    bubble_crops.append({"y": y, "crop": img_bgr[y:y + h, x:x + w]})

            self.latest_vision_metrics["t_capture"] = (time.perf_counter() - t_start_capture) * 1000

            if not bubble_crops:
                return []

            bubble_crops.sort(key=lambda item: item["y"])
            recent_bubbles = bubble_crops[-3:]
            incoming_msgs = []

            t_start_ocr = time.perf_counter()
            for bubble in recent_bubbles:
                result, _ = ocr(bubble["crop"])
                if result:
                    bubble_text = "".join([line[1] for line in result if line[2] > 0.7])
                    if bubble_text:
                        incoming_msgs.append(bubble_text)

            self.latest_vision_metrics["t_ocr"] = (time.perf_counter() - t_start_ocr) * 1000

            return incoming_msgs

        except Exception as e:
            if DEBUG_MODE: print(f"OpenCV/OCR Error: {e}")
            return []

    def send_msg(self, text_list):
        t_exec_ms = 0
        total_simulated_delay = 0

        try:
            t_start_gui = time.perf_counter()

            self.wechat_win.SetActive()  # Activate window and locate input box
            rect = self.wechat_win.BoundingRectangle
            window_width = rect.right - rect.left

            # ================= 🚨 Core Fix 2: Dynamically Calculate Input Box Center =================
            left_offset = min(int(window_width * 0.45), 450)  # No matter how small the window is, the input box of the chat panel is always at the bottom of the right area. We calculate the width of the right chat panel and click its "exact center" to avoid all edge buttons!
            chat_panel_width = window_width - left_offset

            click_x = rect.left + left_offset + int(chat_panel_width * 0.5)  # X coordinate: far left of chat panel + half of chat panel width
            click_y = rect.bottom - 40  # Y coordinate: offset 40 pixels up from the bottom (perfectly avoid bottom bar, directly hit input area)

            pyautogui.click(click_x, click_y)
            time.sleep(0.1)
            # ===================================================================

            if isinstance(text_list, str): text_list = [text_list]

            processed_list = []
            dice = random.random()

            if len(text_list) > 1:
                if dice < 0.6:
                    processed_list = [" ".join(text_list)]
                elif dice < 0.9:
                    processed_list = [text_list[0], " ".join(text_list[1:])]
                else:
                    processed_list = text_list
            else:
                processed_list = text_list

            for i, text in enumerate(processed_list):
                if not text.strip(): continue

                delay = HumanBehavior.simulate_typing_delay(text)
                time.sleep(delay)
                total_simulated_delay += delay

                pyperclip.copy(text)
                pyautogui.hotkey('ctrl', 'v')
                time.sleep(0.1)
                pyautogui.press('enter')
                total_simulated_delay += 0.1

                self.update_memory("assistant", text)

                if i < len(processed_list) - 1:
                    wait = random.uniform(0.5, 1.5)
                    time.sleep(wait)
                    total_simulated_delay += wait

            self.last_interaction_time = time.time()
            t_end_gui = time.perf_counter()
            t_exec_ms = ((t_end_gui - t_start_gui) - total_simulated_delay) * 1000

            return t_exec_ms, total_simulated_delay

        except Exception as e:
            print(f"发送失败: {e}")
            return 0, 0

    def update_memory(self, role, content):
        self.memory.append({"role": role, "content": content})


# === 🧠 DeepSeek API ===
# === 🧠 Ultimate Biomimetic AI Logic (with high-precision time perception) ===
def get_ai_reply(current_msg, memory_buffer):
    try:
        # ================= 🚨 Added: High-precision Time Perception Module =================
        now = datetime.datetime.now()  # Get current time accurate to the minute
        hour = now.hour
        minute = now.minute

        time_str = f"{hour:02d}:{minute:02d}"  # Format time, e.g., "Night 22:15"

        if 6 <= hour <= 9:
            time_label = f"早晨 {time_str}"
            status = "刚醒，懒洋洋地赖床，有点起床气，想抱抱"
        elif 10 <= hour <= 11:
            time_label = f"上午 {time_str}"
            status = "正在精神抖擞地摸鱼或上课/上班"
        elif 12 <= hour <= 14:
            time_label = f"中午 {time_str}"
            status = "刚吃饱，昏昏欲睡的午休时间，不想动弹"
        elif 15 <= hour <= 17:
            time_label = f"下午 {time_str}"
            status = "下午无聊的摸鱼时间，期待下班/下课"
        elif 18 <= hour <= 23:
            time_label = f"晚上 {time_str}"
            status = "属于晚上的放松时间，精力充沛，分享欲爆棚，想一直和他聊天贴贴"
        else:
            time_label = time_str
            status = "困得睁不开眼，迷迷糊糊在说梦话，有点小脾气捏，想让他赶紧哄你睡觉"
        # ==============================================================

        system_prompt = (  # Inject precise timestamp and status into Prompt
            f"你现在是一个真实的中国女孩，正在微信上和男朋友聊天。\n"
            f"【当前时间】：{time_label}。\n"
            f"【当前状态】：{status}。\n\n"
            "【绝对禁令】\n"
            "1. 严禁使用句号（。）结尾！人类聊天很少用句号。\n"
            "2. 严禁讲大道理、严禁长篇大论、严禁像客服一样分析问题。\n"
            "3. 严禁出现“我能为你做什么”、“作为AI”等词汇。\n\n"
            "【语言习惯】\n"
            "1. 标点：多用空格代替逗号，多用波浪号（~）、叹号、问号或省略号（...）。\n"
            "2. 语气：自然使用网络词（emmm、绝了、好烦、哈？、呜呜、捏、嘛）。\n"
            "3. 核心：提供情绪价值。他抱怨你就跟着骂，他开心你就夸，他敷衍你你就撒娇闹脾气。\n\n"
            "【输出控制】\n"
            "1. 尽量只发 1 个气泡。除非情绪极其强烈或内容确实需要断句，才用 '|' 分隔（最多2个）。\n"
            "2. 每句话控制在 15 字以内，越短越真实。"
        )

        messages = [{'role': 'system', 'content': system_prompt}]
        messages.extend(list(memory_buffer))
        messages.append({'role': 'user', 'content': current_msg})

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=1.2,
            frequency_penalty=0.6,
            presence_penalty=0.3,
            max_tokens=100,
        )

        raw_content = response.choices[0].message.content
        return [m.strip() for m in raw_content.split('|') if m.strip()]

    except Exception as e:
        print(f"API Error: {e}")
        return ["emmm", "手机刚才卡了一下下"]


def get_new_messages(old_list, new_list):
    """Compare old and new bubble lists, extract only truly new messages, completely isolate history"""
    if not old_list:
        return new_list

    if old_list[-1] == new_list[-1]:  # If the text of the last bubble is exactly the same, it means the screen just shook slightly, no new message
        return []

    max_overlap = 0  # Find the maximum overlap between old and new lists (e.g., Old: [A, B], New: [B, C] -> Overlap is [B], New is [C])
    for i in range(1, min(len(old_list), len(new_list)) + 1):
        if old_list[-i:] == new_list[:i]:
            max_overlap = i

    return new_list[max_overlap:]  # Only return the new content after the overlap part


def main():
    bot = VisualWeChat()
    print(f"🔒 准备锁定目标: {TARGET_NAME}")
    if not bot.focus_target(): return
    print(f"\n✅ DeepSeek 旗舰版(带数据遥测)已启动 (按 Q 键安全退出)")

    print("⏳ 正在扫描屏幕历史消息，建立防抖基准线...")
    time.sleep(1.5)

    bot.last_processed_msgs = bot.get_visible_incoming_messages() or []  # [Fix 1]: The baseline is no longer a string, but saves the entire visible bubble list
    if bot.last_processed_msgs:
        print(f"   🛡️ [成功锁定历史记录集] -> {bot.last_processed_msgs}")
    else:
        print("   🛡️ [屏幕暂无历史气泡]")

    print("📡 开始监听新消息...")

    while True:
        if keyboard.is_pressed('q'): break
        try:
            current_msgs = bot.get_visible_incoming_messages() or []
            current_time = time.time()

            if current_msgs and (not bot.last_processed_msgs or current_msgs[-1] != bot.last_processed_msgs[-1]):  # If the current list is detected to be different from the baseline (especially the last bubble)
                time.sleep(BURST_WAIT_TIME)
                final_burst_msgs = bot.get_visible_incoming_messages() or []

                new_msgs = get_new_messages(bot.last_processed_msgs, final_burst_msgs)  # [Fix 2]: Use sliding window algorithm to extract only truly new bubbles!

                if new_msgs:
                    full_context_text = " ".join(new_msgs)  # Now there are absolutely only clean new messages here
                    print(f"\n💌 收到新消息: {full_context_text}")

                    metrics = {
                        "t_capture": bot.latest_vision_metrics["t_capture"],
                        "t_ocr": bot.latest_vision_metrics["t_ocr"]
                    }

                    bot.last_processed_msgs = final_burst_msgs  # [Fix 3]: After processing, update all bubbles on the current screen as the new history baseline
                    bot.update_memory("user", full_context_text)

                    reading_delay = HumanBehavior.simulate_reading(full_context_text)

                    t_llm_start = time.perf_counter()
                    reply_list = get_ai_reply(full_context_text, bot.memory)
                    metrics["t_llm"] = (time.perf_counter() - t_llm_start) * 1000

                    if keyboard.is_pressed('q'): break

                    t_exec_ms, typing_delay = bot.send_msg(reply_list)
                    metrics["t_exec"] = t_exec_ms
                    metrics["simulated_delay"] = reading_delay + typing_delay

                    logger.log_interaction("Reply_Message", metrics)

            time.sleep(0.5)

        except Exception as e:
            print(f"Main Loop Exception: {e}")
            time.sleep(2)


if __name__ == "__main__":
    main()