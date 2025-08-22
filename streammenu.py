import streamlit as st
import os
import subprocess

import streamlit as st
import gradio as gt
import streamlit as st
import psutil
from PIL import Image, ImageDraw, ImageFont
import io
import datetime
import urllib.parse
import pywhatkit
from twilio.rest import Client
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import time
import csv
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

filename = "D:\\lw classes\\lw_streamlit_pr\\data.csv"  # replaced your file

sid, token, number, YourNumber,YourEmail,GenApi = None, None, None, None,None,None
with open(filename, mode='r') as file:
    reader = csv.DictReader(file)  # Uses header row automatically
    for row in reader:
        sid = row['sid']
        token = row['token']
        number = row['number']
        YourNumber = row['YourNumber']
        YourEmail = row['YourEmail']
        GenApi = row['GenApi']

# class AllFunction():
    # def ReadRam():
    #     mem = psutil.virtual_memory()
    #     return {"total" :f"{mem.total / (1024 ** 3):.2f}GB","used": f"{mem.used / (1024 ** 3):.2f} GB","free": f"{mem.free / (1024 ** 3):.2f}","usage": f"{mem.percent} %" }
# """

# ****   ****   All Function   ****   ****


# """


# 1. RAM info
def Rram():
    mem = psutil.virtual_memory()
    return f"""
Total: {mem.total / (1024 ** 3):.2f} GB
Used: {mem.used / (1024 ** 3):.2f} GB
Free: {mem.free / (1024 ** 3):.2f} GB
Usage: {mem.percent} %
"""

# 2. Send Email via Gmail web link
def send_email(to, subject, body):
    params = urllib.parse.urlencode({
        'to': to,
        'subject': subject,
        'body': body
    })
    url = f"https://mail.google.com/mail/?view=cm&fs=1&{params}"
    return url

# 3. Schedule WhatsApp message using pywhatkit
def send_whatsapp(number, message, delay_sec):
    future_time = datetime.datetime.now() + datetime.timedelta(seconds=int(delay_sec))
    hour, minute = future_time.hour, future_time.minute
    try:
        pywhatkit.sendwhatmsg(f"+91{number}", message, hour, minute)
        return f"✅ Message scheduled to {number} at {hour}:{minute}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# 4. Create Image
def CreateImg():
    width, height = 800, 600
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    for y in range(height):
        r = int(255 * (y / height))
        g = int(255 * (1 - y / height))
        b = 200
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    import random
    for _ in range(50):
        x0 = random.randint(0, width - 50)
        y0 = random.randint(0, height - 50)
        x1 = x0 + random.randint(20, 80)
        y1 = y0 + random.randint(20, 80)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.ellipse([x0, y0, x1, y1], fill=color, outline="black")

    for _ in range(20):
        x0, y0 = random.randint(0, width), random.randint(0, height)
        x1, y1 = random.randint(0, width), random.randint(0, height)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.line([x0, y0, x1, y1], fill=color, width=3)

    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    draw.text((20, 20), "My Digital Art", font=font, fill="black")
    return image

# 5. Send SMS via Twilio
def send_sms():
    account_sid = sid
    auth_token = token
    twilio_number = number
    recipient_number = YourNumber  # Replace with the actual number
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body="Hello from Python via Twilio 🚀",
        from_=twilio_number,
        to=recipient_number
    )
    return message.sid

# 6. Make Call via Twilio
def Mak_Call():
    account_sid = sid
    auth_token = token
    twilio_number = number
    to_number = YourNumber  # Replace with the actual number
    twiml_url = 'http://demo.twilio.com/docs/voice.xml'
    client = Client(account_sid, auth_token)
    call = client.calls.create(
        to=to_number,
        from_=twilio_number,
        url=twiml_url
    )
    return f"📞 Call initiated: SID {call.sid}"

# 7. Google Search (show top 10 results)
def search_on_google(query):
    urls = []
    titles = []
    for url in search(query, num_results=10):
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.string.strip() if soup.title else url
        except Exception:
            title = url
        titles.append(title)
        urls.append(url)
    return list(zip(titles, urls))

# 8. Download Data from Web (tables/images)
def download_tables(url):
    import pandas as pd
    folder = "tables"
    os.makedirs(folder, exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()
    tables = pd.read_html(response.text)
    saved_files = []
    for i, table in enumerate(tables, 1):
        filename = os.path.join(folder, f"table_{i}.xlsx")
        table.to_excel(filename, index=False)
        saved_files.append(filename)
    return saved_files

def download_images(url):
    from urllib.parse import urljoin, urlparse
    folder = "images"
    os.makedirs(folder, exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    downloaded = []
    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            continue
        img_url = urljoin(url, src)
        parsed = urlparse(img_url)
        filename = os.path.basename(parsed.path)
        if not filename:
            continue
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            continue
        try:
            img_resp = requests.get(img_url)
            img_resp.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(img_resp.content)
            downloaded.append(filepath)
        except:
            pass
    return downloaded


def send_linkedin_message(username, password, recipient_profile_url, message_text):
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.common.by import By
    import time
    
    # 1) create driver correctly
    service = Service(ChromeDriverManager().install())
    driver  = webdriver.Chrome(service=service)
    
    try:
        driver.get("https://www.linkedin.com/login")
        time.sleep(4)

        # 2) log in
        driver.find_element(By.ID, "username").send_keys(username)
        driver.find_element(By.ID, "password").send_keys(password)
        driver.find_element(By.XPATH, "//button[@type='submit']").click()
        time.sleep(40)

        # 3) go to profile & click “Message”
        driver.get(recipient_profile_url)
        time.sleep(4)

        message_btn = driver.find_element(
            By.XPATH, "//button[contains(., 'Message')]"
        )
        message_btn.click()
        time.sleep(4)

        # 4) write & send text
        textbox = driver.find_element(By.XPATH, "//div[@role='textbox']")
        textbox.send_keys(message_text)
        time.sleep(4)
        driver.find_element(By.XPATH, "//button[@type='submit']").click()
        time.sleep(5)
        print("✅  Message sent.")
    except Exception as e:
        print(f"❌  Error: {e}")
    finally:
        time.sleep(5)
        
def post_to_instagram(username, password, message, image_path=None):
    from twilio.rest import Client
    import os
    """
    Posts an image or video to Instagram feed, or a text-based story if no media is provided.

    Args:
        username (str): Instagram username
        password (str): Instagram password
        message (str): Caption for the media or text for story
        image_path (str, optional): Path to the image/video file. If None, a text story is posted.
    """
    cl = Client()
    session_file = f"{username}_session.json"

    try:
        # Load existing session
        if os.path.exists(session_file):
            cl.load_settings(session_file)

        # Login (always attempt fresh login)
        cl.login(username, password)

        # Save session to file
        cl.dump_settings(session_file)

        # Post content
        if image_path:
            extension = image_path.lower().split('.')[-1]
            if extension in ['mp4', 'mov']:
                cl.video_upload(image_path, caption=message)
                print("✅ Video posted successfully!")
            elif extension in ['jpg', 'jpeg', 'png']:
                cl.photo_upload(image_path, caption=message)
                print("✅ Photo posted successfully!")
            else:
                print(f"❌ Unsupported file type: {extension}")
        else:
            # Posting a simple text story (not supported natively without media)
            # Instead, we create a white background image and post as story
            from PIL import Image, ImageDraw, ImageFont
            temp_image = "temp_story.jpg"
            img = Image.new("RGB", (720, 1280), color="white")
            draw = ImageDraw.Draw(img)
            draw.text((50, 600), message, fill="black")
            img.save(temp_image)
            cl.photo_upload_to_story(temp_image)
            os.remove(temp_image)
            print("✅ Text story posted as image!")

    except Exception as e:
        print("⚠️ Login failed. Possible reasons:")
        print("- Invalid or expired session")
        print("- 2FA enabled or suspicious login detected")
        print(f"Details: {e}")
        print(f"❌ Unexpected error: {e}")

    finally:
        password = None  # Clear password for security


def Electricity_Cost():
    st.title("🏠 Electric Fitting Cost Estimator")
    st.markdown("Estimate your house's electric fitting cost based on number of rooms, size, and wire thickness.")

    COST_PER_LIGHT = 150
    COST_PER_FAN = 1200
    COST_PER_SWITCH = 40
    COST_PER_SOCKET = 120
    COST_PER_MCB = 500

    WIRE_COST = {
        "1.0 mm": 20,
        "1.5 mm": 30,
        "2.5 mm": 40,
        "4.0 mm": 60
    }

    st.sidebar.header("🏠 Room Details")
    num_rooms = st.sidebar.number_input("Number of Rooms", min_value=1, value=2)
    num_halls = st.sidebar.number_input("Number of Halls", min_value=0, value=1)
    room_area = st.sidebar.number_input("Average Area of a Room (sq. ft.)", min_value=50, value=120)
    hall_area = st.sidebar.number_input("Average Area of a Hall (sq. ft.)", min_value=100, value=180)
    wire_size = st.sidebar.selectbox("Select Wire Thickness", options=list(WIRE_COST.keys()))

    total_area = (num_rooms * room_area) + (num_halls * hall_area)
    lights = int(total_area // 50)
    fans = int(total_area // 120)
    switches = lights + fans
    sockets = int(total_area // 100)
    mcb = 1
    wiring_length = int((total_area / 100) * 10)

    total_cost = (
        lights * COST_PER_LIGHT +
        fans * COST_PER_FAN +
        switches * COST_PER_SWITCH +
        sockets * COST_PER_SOCKET +
        mcb * COST_PER_MCB +
        wiring_length * WIRE_COST[wire_size]
    )

    st.subheader("📊 Cost Estimation Summary")
    st.write(f"🧱 Total Area: {total_area} sq. ft.")
    st.write(f"🔌 Estimated Lights: {lights}")
    st.write(f"🌀 Estimated Fans: {fans}")
    st.write(f"🎚️ Estimated Switches: {switches}")
    st.write(f"🔌 Estimated Power Sockets: {sockets}")
    st.write(f"🧯 Wire Length Required: {wiring_length} meters ({wire_size})")
    st.write(f"📦 MCB Box: {mcb}")
    st.markdown("---")
    st.success(f"💰 **Total Estimated Cost: ₹{total_cost}**")

    with st.expander("📌 Tips"):
        st.write("- Always keep 10–15% extra wiring for flexibility.")
        st.write("- For kitchen and AC, use 2.5 mm or 4 mm wire.")
        st.write("- Confirm socket placements with an electrician.")


def Mark_Pre():
    st.title("📚 Study Hours vs Marks Predictor")
    st.markdown("Predict your marks based on study hours using linear regression")

    @st.cache_data
    def load_data():
        data = {
            'Hours': [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7],
            'Marks': [21, 47, 27, 75, 30, 20, 88, 60, 81, 25]
        }
        return pd.DataFrame(data)

    df = load_data()
    with st.expander("View Raw Data"):
        st.dataframe(df)

    model = LinearRegression()
    model.fit(df[['Hours']], df['Marks'])

    hours = st.sidebar.number_input("Enter study hours:", min_value=0.5, max_value=24.0, value=5.0, step=0.5)
    if st.sidebar.button("Predict Marks"):
        prediction = model.predict([[hours]])[0]
        st.success(f"Predicted Marks: {prediction:.1f}")

    st.header("Model Information")
    col1, col2 = st.columns(2)
    col1.metric("Coefficient (Slope)", f"{model.coef_[0]:.2f}")
    col2.metric("Intercept", f"{model.intercept_:.2f}")
    st.markdown(f"*Regression Equation:* Marks = {model.coef_[0]:.2f} × Hours + {model.intercept_:.2f}")

    st.header("Data Visualization")
    st.scatter_chart(df.set_index('Hours'))
    x_values = np.linspace(df['Hours'].min(), df['Hours'].max(), 100).reshape(-1, 1)
    y_values = model.predict(x_values)
    st.line_chart(pd.DataFrame({'Predicted Marks': y_values.flatten()}, index=x_values.flatten()))


def PG_RentPr():
    st.title("🏠 PG Price Predictor")
    st.write("Predict the monthly PG price based on room sharing, AC, and food.")

    data = {
        'Persons': [1, 2, 2, 3, 3, 1, 1, 2, 3],
        'AC': [1, 1, 0, 1, 0, 0, 1, 0, 1],
        'Food': [1, 0, 1, 1, 0, 1, 1, 0, 1],
        'Price': [9000, 6500, 5500, 5000, 4000, 8500, 9500, 5200, 4800]
    }
    df = pd.DataFrame(data)
    X = df[['Persons', 'AC', 'Food']]
    y = df['Price']
    model = LinearRegression().fit(X, y)

    st.sidebar.header("📝 Your Room Preferences")
    persons = st.sidebar.selectbox("Number of Persons in Room", [1, 2, 3])
    ac_room = st.sidebar.radio("Room Type", ["AC", "Non-AC"])
    food = st.sidebar.radio("Food Included", ["Yes", "No"])

    ac_flag = 1 if ac_room == "AC" else 0
    food_flag = 1 if food == "Yes" else 0

    predicted_price = model.predict([[persons, ac_flag, food_flag]])[0]

    st.subheader("💰 Estimated Monthly PG Rent")
    st.success(f"For a {persons}-person {'AC' if ac_flag else 'Non-AC'} room with{'out' if not food_flag else ''} food: ₹{predicted_price:.2f}")

    with st.expander("📂 Show Sample Training Data"):
        st.dataframe(df)

    with st.expander("📈 Model Equation"):
        coef = model.coef_
        intercept = model.intercept_
        st.code(f"Price = {coef[0]:.2f} × Persons + {coef[1]:.2f} × AC + {coef[2]:.2f} × Food + {intercept:.2f}")


def Student_guid():
    st.title("Student Career Guidance by Subject")

    guidance = {
        "math": ["Engineer", "Data Scientist", "Mathematician", "Actuary", "Economist"],
        "science": ["Doctor", "Pharmacist", "Research Scientist", "Biotechnologist", "Environmentalist"],
        "physics": ["Mechanical Engineer", "Physicist", "Astronomer", "Robotics Engineer"],
        "chemistry": ["Chemical Engineer", "Pharmacologist", "Forensic Scientist", "Material Scientist"],
        "biology": ["Doctor", "Geneticist", "Zoologist", "Microbiologist"],
        "computer": ["Software Engineer", "Web Developer", "AI/ML Engineer", "Cybersecurity Expert"],
        "english": ["Journalist", "Content Writer", "Teacher", "Editor"],
        "history": ["Historian", "Archaeologist", "Civil Services", "Museum Curator"],
        "geography": ["Geologist", "Urban Planner", "Cartographer", "Environmental Consultant"],
        "commerce": ["CA (Chartered Accountant)", "Banker", "Business Analyst", "Financial Advisor"],
        "arts": ["Designer", "Animator", "Musician", "Fine Artist"]
    }

    subject = st.text_input("Enter your favorite subject:")

    if subject:
        subject_lower = subject.lower()
        if subject_lower in guidance:
            st.write(f"### 📘 Based on your interest in {subject.capitalize()}, you can explore:")
            for career in guidance[subject_lower]:
                st.write(f"- {career}")
        else:
            st.warning("⚠️ Sorry, guidance for this subject is not available.")
            st.info("Try subjects like: Math, Science, Physics, Computer, etc.")


def dockerfile_menu():
    st.header("Dockerfile Commands Menu")

    commands = {
        "FROM": "Specifies the base image to use for the new image.",
        "RUN": "Executes commands in the container during build (e.g., install packages).",
        "CMD": "Sets the default command to run when the container starts.",
        "LABEL": "Adds metadata to the image (like author, version).",
        "EXPOSE": "Declares which ports the container will listen on at runtime.",
        "ENV": "Sets environment variables inside the container.",
        "ADD": "Copies files/folders from source and adds them to the container filesystem (supports URLs).",
        "COPY": "Copies files/folders from source to container filesystem (preferred over ADD if no URLs).",
        "ENTRYPOINT": "Configures a container that will run as an executable.",
        "VOLUME": "Creates a mount point with a specified path to persist data or share between containers.",
        "USER": "Sets the user (or UID) to run subsequent commands and container processes as.",
        "WORKDIR": "Sets the working directory for subsequent commands.",
        "ARG": "Defines build-time variables that users can pass at build time.",
        "ONBUILD": "Adds a trigger instruction to run when the image is used as a base for another build.",
        "HEALTHCHECK": "Defines a command to test container health status.",
        "SHELL": "Overrides the default shell used for RUN commands."
    }

    cmd_choice = st.selectbox("Select a Dockerfile command:", ["-- Select --"] + list(commands.keys()))

    if cmd_choice != "-- Select --":
        st.markdown(f"### {cmd_choice}")
        st.write(commands[cmd_choice])

def Automation_genai():


    import google.generativeai as genai
    import streamlit as st
    
    # Configure your Google API key
    API_KEY = GenApi
    genai.configure(api_key=API_KEY)
    
    # Initialize the Gemini model
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    # Streamlit app
    st.title("🤖 ChatGPT Automation Agent - Career Counselor")
    
    # Chat history stored in session state
    if "history" not in st.session_state:
        st.session_state.history = [
            {
                "role": "user",
                "parts": ["You are a career counselor who helps users choose and understand career options."]
            }
        ]
    
    # User input
    user_input = st.text_input("Ask me anything about careers in AI or related fields:")
    
    if user_input:
        # Append user message to history
        st.session_state.history.append({"role": "user", "parts": [user_input]})
    
        # Start chat with history
        chat = model.start_chat(history=st.session_state.history)
    
        # Send user message and get response
        response = chat.send_message(user_input)
    
        # Append assistant reply to history
        st.session_state.history.append({"role": "assistant", "parts": [response.text]})
    
        # Display assistant response
        st.write(f"**Agent:** {response.text}")
    
    # Show full chat history (optional)
    if st.checkbox("Show full conversation history"):
        for msg in st.session_state.history:
            role = msg["role"].capitalize()
            text = " ".join(msg["parts"])
            st.markdown(f"**{role}:** {text}")


def missing_data_visualizer():
    st.title("🧼 Missing Data Analysis Dashboard")
    
    # Try to load default file if no upload
    default_path = "customers-100.csv"
    
    uploaded_file = st.file_uploader("Upload your CSV file (optional)", type=["csv"])
    
    if uploaded_file:
        datasets = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    elif os.path.exists(default_path):
        datasets = pd.read_csv(default_path)
        st.warning(f"📂 Loaded default file from: `{default_path}`")
    else:
        datasets = None
        st.info("👈 Please upload a CSV file or check local path.")
    
    if datasets is not None:
        st.subheader("Preview of Dataset")
        st.dataframe(datasets.head(5))
    
        st.markdown("---")
        st.subheader("📊 Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", datasets.shape[0])
        with col2:
            st.metric("Total Columns", datasets.shape[1])
    
        total_missing = datasets.isnull().sum().sum()
        percent_missing = (total_missing / (datasets.shape[0] * datasets.shape[1])) * 100
    
        st.write(f"**🔎 Total Missing Values:** {total_missing}")
        st.write(f"**📉 Overall Missing Data (%):** {percent_missing:.2f}%")
    
        st.markdown("---")
        st.subheader("📌 Column-wise Missing Data (%)")
        col_missing = (datasets.isnull().sum() / datasets.shape[0]) * 100
        missing_df = col_missing[col_missing > 0].round(2).to_frame(name="% Missing")
        st.dataframe(missing_df)
    
        st.markdown("---")
        st.subheader("📊 Heatmap of Missing Data")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(datasets.isnull(), cbar=False, cmap="Reds", yticklabels=False)
        st.pyplot(fig)
    else:
        st.stop()



# """


# ****    ****   Actuat Code  ****    ****


# """

st.sidebar.header("Lw Project")
p= st.sidebar.selectbox(label="Project Name",options=["About Me","Project1","Project2","Project3","Project4","Project5"])

if p.__contains__("About Me"):
    import streamlit as st

    # Page config
    st.set_page_config(page_title="About Me", page_icon="😊", layout="wide")
    
    # Title
    st.title("👋 Hi, I'm Rahul Sen")
    st.subheader("B.Tech 4th Year Student | Tech Enthusiast | Lifelong Learner")
    
    # Intro
    st.write("""
    I am currently pursuing my **B.Tech in Computer Science**, now in my 4th year.  
    I’m passionate about technology, problem-solving, and building creative projects.  
    Recently, I’ve been working on:
    - **Generative AI Applications**
    - **Full-Stack Web Development**
    - **Automation Scripts**
    - **Flutter Portfolio Apps**
    """)
    
    # Education
    st.header("🎓 Education")
    st.write("""
    - **B.Tech (Computer Science)** — Currently in 4th Year
    - Ongoing learning in AI, machine learning, and cloud computing
    """)
    
    # Skills
    st.header("💻 Skills")
    skills = ["Python", "JavaScript", "HTML/CSS", "Streamlit", "Flutter", "AI/ML Basics", "APIs Integration", "Automation"]
    st.write(", ".join(skills))
    
    # Projects
    st.header("🚀 Projects")
    st.markdown("""
    1. **Jarvis Desktop Assistant** – A Python + Web-based personal assistant with voice commands, device control, and AI chat.
    2. **Flutter Portfolio App** – Personal app with animated avatar and contact info.
    3. **Generative AI Tools** – Experimenting with ChatGPT, DALL·E, and HuggingChat integration.
    """)
    
    # Contact
    st.header("📬 Contact")
    st.write("📧 Email: rsurendrasen90@gmail.com")
    st.write("🔗 [LinkedIn](https://www.linkedin.com/in/rahul-sain-88a963288)")
    st.write("🐙 [GitHub](https://github.com/RahulRakhi)")
    
    # Footer
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")
    
elif p.__contains__("Project1"):
    app_mode = st.sidebar.radio("Choose your function:",
                                  ("Automation Function","About GenAi"))
    if app_mode.__contains__("Automation Function"):
        Automation_genai()
    elif app_mode.__contains__("About GenAi"):
        import streamlit as st

        # Page configuration
        st.set_page_config(page_title="Introduction to Generative AI", layout="centered")
        
        # Title
        st.title("🤖 Introduction to Generative AI (GenAI)")
        
        # Introduction
        st.markdown("""
        Generative AI (GenAI) is a branch of artificial intelligence that focuses on creating new, original content — such as text, images, music, code, or even videos — rather than just analyzing existing data.
        """)
        
        # How it works
        st.subheader("How it Works")
        st.markdown("""
        It works by learning patterns from large datasets and then generating outputs that are similar in style or structure to what it has learned.
        
        **Examples:**
        - **Text generation** → ChatGPT writing an article or story.
        - **Image generation** → DALL·E creating art from a description.
        - **Music generation** → AI composing new songs.
        - **Code generation** → AI writing programs from prompts.
        """)
        
        # Technology
        st.subheader("Technology Behind GenAI")
        st.markdown("""
        GenAI uses advanced machine learning models, often **large language models (LLMs)** or **diffusion models**, to produce realistic and creative results.
        """)
        
        # Why it matters
        st.subheader("Why it Matters")
        st.markdown("""
        - **Creativity at scale** – It can produce ideas, designs, and solutions faster than humans.  
        - **Automation** – Speeds up content creation in industries like marketing, software, media, and research.  
        - **Personalization** – Can adapt outputs to match individual styles or needs.
        """)
        
        # Tools
        st.subheader("Examples of Generative AI Tools")
        st.markdown("""
        - **ChatGPT** (text)  
        - **DALL·E / Midjourney** (images)  
        - **GitHub Copilot** (code)  
        - **Synthesia** (video)  
        """)
        
        # Footer
        st.caption("🚀 Created with Streamlit")
        
        
elif p.__contains__("Project2"):
    
    st.title("🚀 Cloud Automation using Python")
    app_mode = st.sidebar.radio("Choose your function:",
                                  ("Boto3 Code ","linkedin"))
    if app_mode.__contains__("Automation Function"):
     pipeline_code = """import json
    import boto3
    import logging
    
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    def lamda_handler(event,context):
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        object_key = event['Records'][0]['s3']['object']['key']
        logger.info(f"New image Uploaded: {object_key}")
    
        existing_images = list_images_in_bucket(bucket_name)
    
        loger.info(f" All images in the bucket '{bucket_name}': {existing_images}")
    
        return {
            'statusCode': 200,
            'body': json.dumps(f"Logged recently uploaded image {object_key} and all existing images")
    
        }
    
    def list_images_in_bucket(bucket_name):
        images_list =[]
    
        response = s3.list_objects_v2(Bucket=bucket_name)
    
        if 'Contents' in response:
            for obj in response['Contents']:
                images_list.append(obj['Key'])
            
        return images_list"""
     st.code(pipeline_code, language="Dockerfile")

    elif app_mode.__contains__("linkedin"):
        st.write("")    


elif p.__contains__("Project3"):
    st.title("🚀 Running Apache Inside a Docker Container")
    app_mode = st.sidebar.radio("Choose your function:",
                                  ("Apache Server Dockerfile","flaskapp"))
    if app_mode.__contains__("Apache Server Dockerfile"):
         pipeline_code = """
         # Use the official Apache HTTP Server base image
         FROM httpd:2.4
         
         # Copy your website files into the container
         # Assuming your website files are in a folder named 'public-html' next to the Dockerfile
         COPY  autofillnumber.html /usr/local/apache2/htdocs/
         
         # Expose port 80 to the host
         EXPOSE 80
         
         # The Apache server starts automatically by default in the httpd image, so no CMD needed
         """
         st.code(pipeline_code, language="Dockerfile")
    elif app_mode.__contains__("flaskapp"):
        st.title("🚀 FlaskApp")
        pipeline_code = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <title>Auto-Fill Phone Number</title>
        </head>
        <body>
        
          <h2>Enter Phone Number</h2>
          
          <input type="text" id="phone" placeholder="Enter phone number">
          <button onclick="fillPhoneNumber()">Auto-Fill</button>
        
          <script>
            function fillPhoneNumber() {
              document.getElementById("phone").value = "9999999999";
            }
          </script>
        
        </body>
        </html>

        
        """
        st.components.v1.html(pipeline_code, height=200)
        st.code(pipeline_code, language="Dockerfile")





elif p.__contains__("Project4"):
    app_mode = st.sidebar.radio("Choose your function:",
                                  ("Flask","Docker file","Jenkinfile"))
    if app_mode.__contains__("Docker file"):
        st.title("🚀 DockerFile")
        pipeline_code = """
       # Use an official lightweight Python image
       FROM python:3.9-slim
       
       # Set working directory inside the container
       WORKDIR /app
       
       # Copy requirements.txt and install dependencies
       COPY requirements.txt .
       RUN pip install --no-cache-dir -r requirements.txt
       
       # Copy the rest of your app code
       COPY . .
       
       # Expose port 5000 (Flask default)
       EXPOSE 5000
       
       # Run the Flask app using gunicorn for production
       CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]

        """
        st.code(pipeline_code, language="Dockerfile")
        
    elif app_mode.__contains__("Flask"):
         pipeline_code = """
            from flask import Flask
            
            app = Flask(__name__)
            
            @app.route('/')
            def home():
                return "Hello, welcome to my Flask app! Vimal Sir"
            
            if __name__ == '__main__':
                app.run(host='0.0.0.0', port=5000, debug=True)
            
         """
         st.code(pipeline_code, language="Dockerfile")
        
    elif app_mode.__contains__("Jenkinfile"):
        import streamlit as st
        
        st.set_page_config(page_title="Jenkins Pipeline Viewer", layout="wide")
        
        pipeline_code = """
        pipeline {
            agent any
            
            environment {
                DOCKERHUB_CREDENTIALS = 'dockerhub-creds-id'  // Jenkins stored credentials ID
                DOCKER_IMAGE = 'your-dockerhub-username/my-flask-app'
                IMAGE_TAG = "latest"
            }
            
            stages {
                stage('Checkout Code') {
                    steps {
                        git 'https://github.com/yourusername/your-flask-repo.git'
                    }
                }
                
                stage('Build Docker Image') {
                    steps {
                        script {
                            dockerImage = docker.build("${DOCKER_IMAGE}:${IMAGE_TAG}")
                        }
                    }
                }
                
                stage('Push to Docker Hub') {
                    steps {
                        script {
                            docker.withRegistry('https://registry.hub.docker.com', DOCKERHUB_CREDENTIALS) {
                                dockerImage.push()
                            }
                        }
                    }
                }
                
                stage('Deploy') {
                    steps {
                        // Example deployment: SSH to your server and run docker commands
                        sshagent(['your-ssh-credentials-id']) {
                            sh '''
                            ssh -o StrictHostKeyChecking=no user@your-server << EOF
                                docker pull ${DOCKER_IMAGE}:${IMAGE_TAG}
                                docker stop flask_app_container || true
                                docker rm flask_app_container || true
                                docker run -d --name flask_app_container -p 5000:5000 ${DOCKER_IMAGE}:${IMAGE_TAG}
                            EOF
                            '''
                        }
                    }
                }
            }
        }
        """
        
        # Display nicely formatted code
        st.title("🚀 Jenkins Pipeline Script")
        st.code(pipeline_code, language="groovy")
        


 

elif p.__contains__("Project5"):
    st.sidebar.header("Lw Project")
    r = st.sidebar.radio("Choose your function:",
                                  ("Python","Fulstack","Docker","Ml","DockerFile","GenAi","AWS","Linux"))
    
    # r= st.sidebar.selectbox(label="Project Name",options=["Python","Fulstack","Docker","Ml","DockerFile","GenAi","AWS","Linux"])
    
    
    if r.__contains__("Python"):
        print(r)
        choice = st.selectbox("Choose your action", [
            "Select",
            "1. RAM Info",
            "2. Send WhatsApp Message",
            "3. Send Email",
            "4. Create Image",
            "5. Send SMS (Twilio)",
            "6. Make Phone Call (Twilio)",
            "7. Search on Google",
            "8. Download Data from Web",
            "9. Send LinkedIn Message",
            "10. Post to Instagram"
            
           ])
        
        if choice == "1. RAM Info":
            st.subheader("RAM Info")
            st.text(Rram())
        
        elif choice == "2. Send WhatsApp Message":
            st.subheader("Send WhatsApp Message")
            number = st.text_input("Phone number (without country code):")
            message = st.text_area("Message:")
            delay_sec = st.number_input("Delay in seconds:", min_value=0, value=5)
            if st.button("Send WhatsApp Message"):
                if number and message:
                    result = send_whatsapp(number, message, delay_sec)
                    st.success(result)
                else:
                    st.error("Please enter both phone number and message.")
        
        elif choice == "3. Send Email":
            st.subheader("Send Email via Gmail")
            to = st.text_input("To (email address):")
            subject = st.text_input("Subject:")
            body = st.text_area("Body:")
            if st.button("Open Gmail Compose"):
                if to and subject and body:
                    url = send_email(to, subject, body)
                    st.markdown(f"[Open Gmail Compose]({url})", unsafe_allow_html=True)
                else:
                    st.error("Please fill all fields.")
        
        elif choice == "4. Create Image":
            st.subheader("Create Digital Art Image")
            if st.button("Generate Image"):
                img = CreateImg()
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                st.image(buf.getvalue(), caption="Generated Digital Art")
        
        elif choice == "5. Send SMS (Twilio)":
            st.subheader("Send SMS via Twilio")
            if st.button("Send SMS"):
                try:
                    sid = send_sms()
                    st.success(f"SMS sent! SID: {sid}")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif choice == "6. Make Phone Call (Twilio)":
            st.subheader("Make Phone Call via Twilio")
            if st.button("Make Call"):
                try:
                    result = Mak_Call()
                    st.success(result)
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif choice == "7. Search on Google":
            st.subheader("Google Search")
            query = st.text_input("Enter search query:")
            if st.button("Search"):
                if query:
                    results = search_on_google(query)
                    for i, (title, url) in enumerate(results, 1):
                        st.markdown(f"{i}. [{title}]({url})")
                else:
                    st.error("Please enter a search query.")
        
        
        elif choice == "8. Download Data from Web":
            st.subheader("Download Tables or Images from Web")
            url = st.text_input("Enter URL:", "https://en.wikipedia.org/wiki/Krishna")
            data_choice = st.radio("What to download?", ("Tables", "Images"))
            if st.button("Download"):
                if url:
                    if data_choice == "Tables":
                        try:
                            files = download_tables(url)
                            if files:
                                for f in files:
                                    st.success(f"Saved table: {f}")
                            else:
                                st.info("No tables found.")
                        except Exception as e:
                            st.error(f"Error downloading tables: {e}")
                    else:
                        try:
                            files = download_images(url)
                            if files:
                                for f in files:
                                    st.success(f"Saved image: {f}")
                            else:
                                st.info("No images found or downloaded.")
                        except Exception as e:
                            st.error(f"Error downloading images: {e}")
                else:
                    st.error("Please enter a valid URL.")
        
        
        if choice == "9. Send LinkedIn Message":
            st.header("Send LinkedIn Message")
            username = st.text_input("LinkedIn Username (email)")
            password = st.text_input("LinkedIn Password", type="password")
            recipient_url = st.text_input("Recipient LinkedIn Profile URL")
            message_text = st.text_area("Message Text")
        
            if st.button("Send Message"):
                if username and password and recipient_url and message_text:
                    with st.spinner("Sending LinkedIn message... This may take some time. Please do not close the browser."):
                        send_linkedin_message(username, password, recipient_url, message_text)
                else:
                    st.error("Please fill all fields.")
        
        elif choice == "10. Post to Instagram":
            st.header("Post to Instagram")
            ig_username = st.text_input("Instagram Username")
            ig_password = st.text_input("Instagram Password", type="password")
            ig_message = st.text_area("Caption / Story Text")
            ig_image = st.file_uploader("Upload Image or Video (optional)", type=['jpg', 'jpeg', 'png', 'mp4', 'mov'])
        
            if st.button("Post to Instagram"):
                if ig_username and ig_password and ig_message:
                    image_path = None
                    if ig_image:
                        # Save to temp file
                        temp_file_path = f"temp_{ig_image.name}"
                        with open(temp_file_path, "wb") as f:
                            f.write(ig_image.getbuffer())
                        image_path = temp_file_path
                    with st.spinner("Posting to Instagram..."):
                        post_to_instagram(ig_username, ig_password, ig_message, image_path)
                    if image_path:
                        os.remove(image_path)
                else:
                    st.error("Please fill username, password, and message at least.")
    
    elif r.__contains__("Ml"):
        app_mode = st.sidebar.radio("Choose your function:",
                                  ("Missing Data Visualizer", "Electricity Cost Estimator", "Marks Predictor", "PG Rent Predictor", "Student Career Guidance"))
    
        if app_mode == "Missing Data Visualizer":
            missing_data_visualizer()
        if app_mode == "Electricity Cost Estimator":
            Electricity_Cost()
        elif app_mode == "Marks Predictor":
            Mark_Pre()
        elif app_mode == "PG Rent Predictor":
           PG_RentPr()
        elif app_mode == "Student Career Guidance":
           Student_guid()
    
    
    elif r.__contains__("Linux"):
        
        def run_command(cmd):
            try:
                # Run command and decode output
                output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
            except subprocess.CalledProcessError as e:
                output = f"Error:\n{e.output}"
            return output
        
        choice = st.selectbox(
            "Choose a Linux command to run:",
            options=[
                "1. Show current date and time (date)",
                "2. Show current directory (pwd)",
                "3. List files and directories (ls)",
                "4. Show disk usage (df -h)",
                "5. Show memory usage (free -h)",
                "6. Show running processes (ps aux)",
                "7. Show current user (whoami)",
                "8. Ping a host",
                "9. Check network interfaces (ip a)",
                "10. Show system uptime (uptime)",
                "11. Show logged-in users (w)",
                "12. Show environment variables (printenv)",
                "13. Show CPU info (lscpu)",
                "14. Show kernel version (uname -r)",
                "15. Show active network connections (ss -tuln)",
                "16. Show open files by a process (lsof -p PID)",
                "17. Display contents of a file (cat)",
                "18. Search text in files (grep)",
                "19. Show disk partition info (lsblk)",
                "20. Show mounted filesystems (mount)",
                "21. Show top 10 processes by memory usage",
                "22. Show last 10 system logs (tail /var/log/syslog)",
                "23. Check listening ports (netstat -tuln)",
                "24. Show active services (systemctl list-units --type=service)",
                "25. Show firewall status (sudo ufw status)",
                "26. Show current directory size (du -sh)",
                "27. Show groups of current user (groups)",
                "28. Show IP routing table (ip route)",
                "29. Show disk inode usage (df -i)",
                "30. Exit"
            ]
        )
        
        if choice == "30. Exit":
            st.write("Exiting... Goodbye!")
        else:
            # Map choices to commands
            if choice.startswith("8"):
                host = st.text_input("Enter host to ping (e.g., google.com):")
                if st.button("Run Command") and host:
                    cmd = f"ping -c 4 {host}"
                    st.text(run_command(cmd))
            elif choice.startswith("16"):
                pid = st.text_input("Enter process ID (PID):")
                if st.button("Run Command") and pid:
                    cmd = f"lsof -p {pid}"
                    st.text(run_command(cmd))
            elif choice.startswith("17"):
                filename = st.text_input("Enter filename to display:")
                if st.button("Run Command") and filename:
                    cmd = f"cat {filename}"
                    st.text(run_command(cmd))
            elif choice.startswith("18"):
                text = st.text_input("Enter text to search for:")
                filename = st.text_input("Enter filename to search in:")
                if st.button("Run Command") and text and filename:
                    cmd = f"grep '{text}' {filename}"
                    st.text(run_command(cmd))
            else:
                commands = {
                    "1": "date",
                    "2": "pwd",
                    "3": "ls -la",
                    "4": "df -h",
                    "5": "free -h",
                    "6": "ps aux",
                    "7": "whoami",
                    "9": "ip a",
                    "10": "uptime",
                    "11": "w",
                    "12": "printenv",
                    "13": "lscpu",
                    "14": "uname -r",
                    "15": "ss -tuln",
                    "19": "lsblk",
                    "20": "mount",
                    "21": "ps aux --sort=-%mem | head -n 11",
                    "22": "tail -n 10 /var/log/syslog",
                    "23": "netstat -tuln",
                    "24": "systemctl list-units --type=service",
                    "25": "sudo ufw status",
                    "26": "du -sh .",
                    "27": "groups",
                    "28": "ip route",
                    "29": "df -i",
                }
        
                key = choice.split(".")[0]
                cmd = commands.get(key)
                if st.button("Run Command") and cmd:
                    st.text(run_command(cmd))
    
    elif r.__contains__("Docker"):
        def run_command(cmd):
            try:
                output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
            except subprocess.CalledProcessError as e:
                output = f"Error:\n{e.output}"
            return output
        
        choice = st.selectbox(
            "Select a Docker command to run:",
            options=[
                "1. Show Docker version",
                "2. Show running containers",
                "3. Show all containers",
                "4. Show all images",
                "5. Pull an image",
                "6. Run a container (interactive mode)",
                "7. Run a container (detached mode)",
                "8. Stop a container",
                "9. Start a container",
                "10. Restart a container",
                "11. Remove a container",
                "12. Remove all stopped containers",
                "13. Remove an image",
                "14. Remove all unused images",
                "15. Show container logs",
                "16. Execute a command inside a container",
                "17. Inspect a container",
                "18. Show container stats",
                "19. Copy file from host to container",
                "20. Copy file from container to host",
                "21. Save image to tar file",
                "22. Load image from tar file",
                "23. Show Docker info",
                "24. Exit"
            ]
        )
        
        if choice == "24. Exit":
            st.write("Exiting Docker Menu... Goodbye!")
        else:
            run = False
            cmd = ""
        
            if choice == "5. Pull an image":
                img = st.text_input("Enter image name (e.g., ubuntu:latest):")
                run = st.button("Run Command") and img.strip() != ""
                cmd = f"docker pull {img}"
        
            elif choice == "6. Run a container (interactive mode)":
                img = st.text_input("Enter image name:")
                run = st.button("Run Command") and img.strip() != ""
                cmd = f"docker run -it {img} /bin/bash"
        
            elif choice == "7. Run a container (detached mode)":
                img = st.text_input("Enter image name:")
                run = st.button("Run Command") and img.strip() != ""
                cmd = f"docker run -d {img}"
        
            elif choice in ["8. Stop a container", "9. Start a container", "10. Restart a container",
                            "11. Remove a container", "15. Show container logs", "17. Inspect a container"]:
                cid = st.text_input("Enter container ID or name:")
                run = st.button("Run Command") and cid.strip() != ""
                cmds = {
                    "8. Stop a container": f"docker stop {cid}",
                    "9. Start a container": f"docker start {cid}",
                    "10. Restart a container": f"docker restart {cid}",
                    "11. Remove a container": f"docker rm {cid}",
                    "15. Show container logs": f"docker logs {cid}",
                    "17. Inspect a container": f"docker inspect {cid}",
                }
                cmd = cmds[choice]
        
            elif choice == "12. Remove all stopped containers":
                run = st.button("Run Command")
                cmd = "docker container prune -f"
        
            elif choice == "13. Remove an image":
                img = st.text_input("Enter image name or ID:")
                run = st.button("Run Command") and img.strip() != ""
                cmd = f"docker rmi {img}"
        
            elif choice == "14. Remove all unused images":
                run = st.button("Run Command")
                cmd = "docker image prune -a -f"
        
            elif choice == "16. Execute a command inside a container":
                cid = st.text_input("Enter container ID or name:")
                cmd_in = st.text_input("Enter command to run inside container:")
                run = st.button("Run Command") and cid.strip() != "" and cmd_in.strip() != ""
                cmd = f"docker exec -it {cid} {cmd_in}"
        
            elif choice == "18. Show container stats":
                run = st.button("Run Command")
                cmd = "docker stats --no-stream"
        
            elif choice == "19. Copy file from host to container":
                src = st.text_input("Enter source file path (host):")
                cid = st.text_input("Enter container ID or name:")
                dest = st.text_input("Enter destination path (container):")
                run = st.button("Run Command") and src.strip() != "" and cid.strip() != "" and dest.strip() != ""
                cmd = f"docker cp {src} {cid}:{dest}"
        
            elif choice == "20. Copy file from container to host":
                cid = st.text_input("Enter container ID or name:")
                src = st.text_input("Enter source file path (container):")
                dest = st.text_input("Enter destination path (host):")
                run = st.button("Run Command") and cid.strip() != "" and src.strip() != "" and dest.strip() != ""
                cmd = f"docker cp {cid}:{src} {dest}"
        
            elif choice == "21. Save image to tar file":
                img = st.text_input("Enter image name:")
                tarfile = st.text_input("Enter tar file name:")
                run = st.button("Run Command") and img.strip() != "" and tarfile.strip() != ""
                cmd = f"docker save {img} -o {tarfile}"
        
            elif choice == "22. Load image from tar file":
                tarfile = st.text_input("Enter tar file path:")
                run = st.button("Run Command") and tarfile.strip() != ""
                cmd = f"docker load -i {tarfile}"
        
            elif choice == "23. Show Docker info":
                run = st.button("Run Command")
                cmd = "docker info"
        
            else:
                st.warning("Feature not implemented or invalid choice.")
        
            if run and cmd:
                st.text_area("Command Output:", run_command(cmd), height=400)
          
    elif r.__contains__("DockerFile"):
        dockerfile_menu()
    
    elif r.__contains__("GenAi"):
        import google.generativeai as genai
        
        # ✅ Set your Gemini API key here securely
        genai.configure(api_key=GenApi)  # ← Your key
        
        # Initialize the Gemini model
        model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
        
        # Streamlit UI setup
        st.set_page_config(page_title="Gemini Meeting Assistant", layout="wide")
        st.title("🤖 Gemini Meeting Assistant")
        st.markdown("Upload a meeting transcript to generate summaries and ask questions.")
        
        # File uploader
        uploaded_file = st.file_uploader("📄 Upload Transcript File (.txt)", type=["txt"])
        
        if uploaded_file:
            # Read uploaded transcript
            transcript_text = uploaded_file.read().decode("utf-8")
        
            st.subheader("📜 Transcript Preview")
            st.text_area("Transcript Content", transcript_text, height=250)
        
            # Generate Summary
            if st.button("📝 Generate Summary"):
                with st.spinner("Generating summary with Gemini..."):
                    response = model.generate_content(
                        f"Summarize the following transcript in 5 bullet points:\n{transcript_text}"
                    )
                    st.subheader("✅ Summary")
                    st.write(response.text)
        
            # Q&A Section
            user_question = st.text_input("❓ Ask something about the meeting:")
            if user_question:
                with st.spinner("Gemini is thinking..."):
                    prompt = f"Here is a transcript:\n{transcript_text}\n\nAnswer this: {user_question}"
                    response = model.generate_content(prompt)
                    st.subheader("💬 Gemini's Answer")
                    st.write(response.text)
        
            # Download Full Summary
            if st.button("📥 Download Full Summary"):
                response = model.generate_content(
                    f"Provide a detailed summary with action items and dates from the transcript:\n{transcript_text}"
                )
                summary = response.text
                st.download_button("Download as TXT", summary, file_name="meeting_summary.txt")
        else:
            st.info("Please upload a .txt transcript file to start.")
    
    elif r.__contains__("Fulstack"):
        
        
        import http.server
        import socketserver
        import webbrowser
        import os
        
        PORT = 8000
        
        # Change working directory to the directory containing index.html
        os.chdir(os.path.dirname(os.path.abspath("D:\lw classes\FinalProjects\projectLw\lw\index.html")))
        
        Handler = http.server.SimpleHTTPRequestHandler
        
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Serving at http://localhost:{PORT}")
            webbrowser.open(f"http://localhost:{PORT}/index.html")
            httpd.serve_forever()
        
    
    
    
    elif r.__contains__("AWS"):
        import streamlit as st
        import boto3
        import csv
        import time
        
        # --- AWS SESSION SETUP ---
        def get_aws_session():
            with open(r'D:\lw classes\lw_streamlit_pr\rootkey.csv', 'r') as file:
                reader = csv.reader(file)
                next(reader)  # skip header
                keys = next(reader)
            return boto3.Session(
                aws_access_key_id=keys[0],
                aws_secret_access_key=keys[1],
            )
        
        # --- INSTANCE MANAGEMENT ---
        def launch_instance():
            ec2 = session.resource('ec2', region_name='ap-southeast-1')
            response = ec2.create_instances(
                ImageId='ami-02c7683e4ca3ebf58',
                InstanceType='t2.micro',
                MinCount=1,
                MaxCount=1,
            )
            instance_id = response[0].instance_id
            return instance_id
        
        def terminate_instance(instance_id):
            ec2 = session.resource('ec2', region_name='ap-southeast-1')
            instance = ec2.Instance(instance_id)
            instance.terminate()
            return f"Terminated EC2 Instance: {instance_id}"
        
        # --- STREAMLIT UI ---
        st.set_page_config(page_title="AWS EC2 Controller", layout="centered")
        st.title("🚀 AWS EC2 Instance Controller")
        
        session = get_aws_session()
        instance_id = st.session_state.get("instance_id", None)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Launch EC2 Instance"):
                instance_id = launch_instance()
                st.session_state["instance_id"] = instance_id
                st.success(f"✅ Launched Instance: {instance_id}")
        
        with col2:
            if st.button("Terminate EC2 Instance"):
                if instance_id:
                    msg = terminate_instance(instance_id)
                    st.success(msg)
                    st.session_state["instance_id"] = None
                else:
                    st.warning("⚠ No running instance to terminate!")
        
        if instance_id:
            st.info(f"Current Active Instance: {instance_id}")
        else:
            st.info("No active instance.")


