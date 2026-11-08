# Banking4all
Sheridan Datathon 2026

## Inspiration
Most financial tools assume life happens in a straight line. They look at median income and median spending, subtract the two, and call the result "financial health." But real life isn't a median, it's volatile. A sudden job loss, an unexpected furnace repair, or a 15% spike in rent can instantly unravel a household.
We wanted to build something that moves beyond sterile averages to uncover the real, messy truth of financial resilience. We were inspired to bridge the gap between hard data science and human empathy creating a tool that doesn't just tell a customer they're going to run out of money, but actually generates a personalized, algorithmic lifeboat to help them survive the shock.

## What it does
Banking4all is an interactive financial stress-testing platform. It operates in three main steps:
1. Know the Customer: It instantly analyzes transaction histories to profile a customer's typical cash flow, fixed costs, and available buffer.   
2. Take a Shock: Through an interactive dashboard, users can inject macroeconomic or personal shocks such as an Employment Insurance (EI) job loss retaining only 55% of income, or a sudden $1,500 repair. The engine instantly recalculates their runway in real-time.   
3. Make a Plan: Instead of just flashing a warning, the engine algorithms pinpoint exactly which flexible spending categories the customer can realistically trim, creating a month-by-month backup plan. Finally, it uses the Gemini API to generate a professional, empathetic summary of the customer's new reality. 

## How we built it
- The Core Engine (engine.py): Built in Python with Pandas and NumPy, this strictly deterministic engine computes the math. It calculates the runway using the official competition formula.
- Data Science & ML (Datathon.ipynb): We used Scikit-Learn to scale our logic across all 4,500 accounts. We built a Gradient Boosting model to predict loan defaults and utilized K-Means clustering to segment the customer base into three distinct behavioral personas.   
- The UI (app.py): We used Streamlit to rapidly scaffold an interactive frontend directly inside our development environment, utilizing data-caching to ensure the sliders and charts update instantly.
- Generative AI: We integrated Google's gemini-1.5-flash model via the google-generativeai SDK to ingest the raw financial outputs and translate them into a human-readable coaching blurb.

## Challenges we ran into
- The Data Leakage Trap: When predicting loan defaults, our initial models scored an unrealistic AUC of 0.98+. We realized the features were computed over the whole history, meaning the model was "cheating" by looking at months after the loan already went bad. We had to restructure the pipeline to strictly use a pre-loan "honest" dataset, which still yielded a strong AUC of 0.836.   
- Hosting Hurdles: Prototyping a Streamlit web app with dynamic CSS inside a cloud notebook environment caused severe port-forwarding and WebSocket issues. We had to bypass standard tunneling services and utilize native kernel proxying with CORS and XSRF protection disabled to get the app running smoothly for the demo.
## Accomplishments that we're proud of
We are incredibly proud of discovering Hidden Fragility.Because the standard runway formula relies on medians, highly volatile accounts can mathematically appear "safe." We built a Monte Carlo simulator to bootstrap 500 alternate futures per customer, randomly drawing from their own historical transaction months. This revealed that 844 customers (almost 19% of the base) who look safe on paper actually fail in $\ge 25\%$ of simulated futures because their month-to-month variance (Coefficient of Variation, $CV = \sigma / \mu$) is too chaotic. 
## What we learned
- Geography is not Destiny: We hypothesized that local district economies (unemployment rates, average salaries) would impact resilience. We ran the Spearman correlation and got ~0.01. We learned that financial resilience is driven almost entirely by individual household spending habits, not postal codes.   
- Averages Hide Struggles: You cannot predict a customer's breaking point without looking at the variance of their habits.
- The Power of "Vibe-Coding": Iteratively pairing strict data science pipelines with rapid Streamlit/LLM prototyping allowed us to transform 100MB of raw CSV data into a polished, deployable product in a matter of hours. 

## What's next for Banking4all
Agentic Financial Coaching: Upgrading the Gemini integration from generating static summaries to a fully conversational RAG (Retrieval-Augmented Generation) agent that can answer specific customer questions about their past transactions.
