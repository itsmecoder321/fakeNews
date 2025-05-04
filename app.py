import streamlit as st
import pickle
import string
import nltk
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import difflib
from newspaper import Article

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

true_df = pd.read_csv("true.csv")
fake_df = pd.read_csv("fake.csv")
true_df['label'] = 0  # 0 = real
fake_df['label'] = 1  # 1 = fake

combined_df = pd.concat([true_df, fake_df], ignore_index=True)
combined_df['cleaned_text'] = combined_df['text'].apply(lambda x: clean_text(str(x)))
original_texts = set(combined_df['cleaned_text'].values)

if "original_news" not in st.session_state:
    st.session_state.original_news = ""

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

def highlight_changes(original_text, changed_text):
    matcher = difflib.SequenceMatcher(None, original_text.split(), changed_text.split())
    highlighted = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace' or tag == 'insert':
            highlighted.append(f'<span style="background-color: red;">{" ".join(changed_text.split()[j1:j2])}</span>')
        else:
            highlighted.append(" ".join(changed_text.split()[j1:j2]))
    return " ".join(highlighted)


st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("Fake News Detection & Changed News Tracker")
st.subheader("Analyze news for authenticity and track edits.")


st.subheader("Enter a News URL to Auto-Fill the Text")
url = st.text_input("Enter News Article URL")

if st.button("Fetch News from URL"):
    if url.strip() == "":
        st.error("Please enter a URL.")
    else:
        try:
            article = Article(url)
            article.download()
            article.parse()
            st.session_state.original_news = article.text
            st.success("✅ Article content loaded successfully!")
        except Exception as e:
            st.error(f"Failed to fetch article. Error: {str(e)}")

st.subheader("Try Sample News Articles from Used Dataset")

real_news_samples = [
    "LONDON (Reuters) - British aid minister Priti Patel will resign rather than be sacked by Prime Minister Theresa May, the BBC s political editor quoted an unnamed source as saying on Wednesday. Patel was meeting May on Wednesday to answer questions over undisclosed meetings with Israeli officials. ",
    "WASHINGTON (Reuters) - The chairman of the U.S. House of Representatives Appropriations Committee introduced a bill on Monday to provide $81 billion in emergency aid for recent hurricanes and wildfires. The legislation includes $27.6 billion for the Federal Emergency Management Agency and $26.1 billion for community development block grants, Representative Rodney Frelinghuysen said in a statement. President Donald Trump had requested $44 billion last month, which was widely criticized by lawmakers as being insufficient. ",
    "NAIROBI (Reuters) - South Sudan is hiking fees for humanitarians and blocking them from reaching hungry families, even as the oil-rich country appeals for nearly 2 billion dollars to help avert starvation amid a civil war, five aid groups told Reuters. The government and the United Nations announced on Wednesday that South Sudan needs $1.7 billion in aid next year to help 6 million people   half its population   cope with the effects of war, hunger and economic decline. But aid groups said bureaucracy, violence and rocketing government fees were stopping their work, despite a promise from President Salva Kiir to allow unhindered access after the United States threatened to pull support to the government in October. All the aid workers spoke on condition of anonymity, citing fear of expulsion from the country.  Alain Noudehou, the U.N. s top humanitarian official in the country, said the increased fees are a major concern.  [It] will take away from the resources we have to address the crisis,  Noudehou, humanitarian coordinator for the U.N. Mission in South Sudan (UNMISS), said on Wednesday.  Juba announced plans in March to charge each foreign aid worker $10,000 per annual permit but later dropped them. It revised the fees steeply upwards last month, however, requiring some foreign aid workers to pay $4,000 for a permit   16 times the old rate.  At least two aid groups have paid, they told Reuters on condition of anonymity.  Humanitarian Affairs Minister Hussein Mar Nyout said on Wednesday he had received many complaints over the new fees and restrictions on travel for some aid workers.   This is not in the spirit of the president and we are going to implement the order of the president,  he said in response to questions at a news conference.                 About one-third of South Sudan s 12 million population have fled their homes since the civil war began in 2013, two years after it won independence from Sudan. The United Nations describes the violence as ethnic cleansing. Earlier this year, pockets of the country plunged briefly into famine.  The economy has nosedived, there is hyperinflation, and the government is unable to pay civil servants and soldiers because oil production has collapsed and official corruption is rampant.      The confusion over permits delays aid, organizations said.    Nobody understands who is giving directives and who is supposed to implement,  said the head of one international aid group in South Sudan. He said customs have seized his organization s IT equipment, despite an import tax waiver for aid groups. Last week, a team of doctors said they were denied permission to travel outside the capital because they had not received work permits they had paid for.  Another aid group said it is unable to bring in foreign  medical staff to complete a government-approved project because authorities said even a consultant visiting South Sudan for a week had to pay $4,000 for a permit that takes months to obtain.  The first issue is the inherent absurdity and impracticality of the rules,  said an employee of the aid group.  The second is that laws are being tried inconsistently by four government agencies that are at loggerheads.  South Sudan expelled the Norwegian Refugee Council s country director last year, while some 28 aid workers have been killed this year, with nine shot dead in November alone, according to the United Nations.   UNMISS staff are exempt from the work permit requirement but the government has forced some contractors to pay, in violation of an agreement with the U.N., an UNMISS spokeswoman said.  Juba is not honoring a similar treaty exempting aid agencies receiving U.S. funding, a Western diplomat said. "
]

fake_news_samples = [
    "The number of cases of cops brutalizing and killing people of color seems to see no end. Now, we have another case that needs to be shared far and wide. An Alabama woman by the name of Angela Williams shared a graphic photo of her son, lying in a hospital bed with a beaten and fractured face, on Facebook. It needs to be shared far and wide, because this is unacceptable.It is unclear why Williams  son was in police custody or what sort of altercation resulted in his arrest, but when you see the photo you will realize that these details matter not. Cops are not supposed to beat and brutalize those in their custody. In the post you are about to see, Ms. Williams expresses her hope that the cops had their body cameras on while they were beating her son, but I think we all know that there will be some kind of convenient  malfunction  to explain away the lack of existence of dash or body camera footage of what was clearly a brutal beating. Hell, it could even be described as attempted murder. Something tells me that this young man will never be the same. Without further ado, here is what Troy, Alabama s finest decided was appropriate treatment of Angela Williams  son:No matter what the perceived crime of this young man might be, this is completely unacceptable. The cops who did this need to rot in jail for a long, long time   but what you wanna bet they get a paid vacation while the force  investigates  itself, only to have the officers returned to duty posthaste?This, folks, is why we say BLACK LIVES MATTER. No way in hell would this have happened if Angela Williams  son had been white. Please share far and wide, and stay tuned to Addicting Info for further updates.Featured image via David McNew/Stringer/Getty Images",
    "It almost seems like Donald Trump is trolling America at this point. In the beginning, when he tried to gaslight the country by insisting that the crowd at his inauguration was the biggest ever   or that it was even close to the last couple of inaugurations   we all kind of scratched our heads and wondered what kind of bullshit he was playing at. Then when he started appointing people to positions they had no business being in, we started to worry that this was going to go much worse than we had expected.After 11 months of Donald Trump pulling the rhetorical equivalent of whipping his dick out and slapping it on every table he gets near, I think it s time we address what s happening: Dude is a straight-up troll. He gets pleasure out of making other people uncomfortable or even seeing them in distress. He actively thinks up ways to piss off people he doesn t like.Let s set aside just for a moment the fact that that s the least presidential  behavior anyone s ever heard of   it s dangerous.His latest stunt is one of the grossest yet.Everyone is, by now, used to Trump not talking about things he doesn t want to talk about, and making a huge deal out of things that not many people care about. So it wasn t a huge surprise when the president didn t discuss the Sandy Hook shooting of 2012 on the fifth anniversary of that tragic event. What was a huge surprise was that he not only consciously decided not to invite the victims  families to the White House Christmas party this year   as they have been invited every year since the massacre took place, along with others who share those concerns.In each of the past 4 years, President Obama invited gun violence prevention activists, gun violence survivors (including the Sandy Hook families) and supportive lawmakers to his Christmas party. Zero gun lobbyists were in attendance. pic.twitter.com/QePW9FtbSh  Shannon Watts (@shannonrwatts) December 15, 2017The last sentence of that tweet is important, because that s exactly who Donald Trump did invite to the White House Christmas party. Instead of victims. On the anniversary day.Yesterday was the 5 year mark of the mass shooting at Sandy Hook School, which went unacknowledged by the President. On the same day, he hosted a White House Christmas party to which he invited @NRA CEO Wayne LaPierre. Here he is at the party with @DanPatrick. pic.twitter.com/mUbKCIWGxB  Shannon Watts (@shannonrwatts) December 15, 2017Wayne LaPierre is the man who, in response to the Sandy Hook massacre, finally issued a statement that blamed gun violence on music, movies, and video games, and culminated with perhaps the greatest bit of irony any man has ever unintentionally conceived of: Isn t fantasizing about killing people as a way to get your kicks really the filthiest form of pornography? Yes. Yes, it is, Wayne.Anyway, Happy Holidays Merry Christmas from Donald Trump, everyone!Featured image via Alex Wong/Getty Images",
    "Donald Trump s current deputy national security adviser K.T. McFarland, a former Fox News personality, K. T. McFarland admitted in an email to a colleague during the 2016 presidential transition to Russia throwing the election to Trump. The leaked email was written just weeks before Trump s inauguration and it states that sanctions would make it difficult to ease relations with Russia,  which has just thrown the U.S.A. election to him. The New York Times reports:But emails among top transition officials, provided or described to The New York Times, suggest that Mr. Flynn was far from a rogue actor. In fact, the emails, coupled with interviews and court documents filed on Friday, showed that Mr. Flynn was in close touch with other senior members of the Trump transition team both before and after he spoke with the Russian ambassador, Sergey I. Kislyak, about American sanctions against Russia.A White House lawyer tried to explain McFarland s email to the The Times by saying that she was referring to the Democrats  portrayal of the election. That doesn t make any sense, by the way.McFarland wrote the email to Thomas P. Bossert, who currently serves as Trump s homeland security adviser, then he forwarded it to future National Security Advisor Michael Flynn (now indicted), future Chief of Staff Reince Priebus, future senior strategist Stephen Bannon, and future press secretary Sean Spicer, the Daily Beast reports.With all the pearl-clutching we witnessed from conservatives about Hillary Clinton s emails, you d think they wouldn t be sending messages about Russia throwing the election to Trump.This past March, John Oliver, the host of the HBO comedy show Last Week Tonight started a segment called  Stupid Watergate,  which he described as  a scandal with all the potential ramifications of Watergate, but where everyone involved is stupid and bad at everything. Nailed it!Photo by Chip Somodevilla/Getty Images."
]

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 🟢 Real News Samples")
    for i, news in enumerate(real_news_samples):
        if st.button(f"Real Sample {i+1}", key=f"real_{i}"):
            st.session_state.input_text = news

with col2:
    st.markdown("#### 🔴 Fake News Samples")
    for i, news in enumerate(fake_news_samples):
        if st.button(f"Fake Sample {i+1}", key=f"fake_{i}"):
            st.session_state.input_text = news

st.subheader("Step 1: Real or Fake News Detection")
input_text = st.text_area("Enter News Text Here:", value=st.session_state.input_text, height=200)
st.session_state.input_text = input_text

if st.button("Check Real/Fake"):
    if input_text.strip() == "":
        st.error("Please enter some text.")
    else:
        cleaned = clean_text(input_text)
        if cleaned not in original_texts:
            st.info("ℹ️ This news was not found in the original training dataset. Prediction is not available.")
        else:
            vec = vectorizer.transform([cleaned]).toarray()
            prediction = model.predict(vec)[0]
            confidence = model.predict_proba(vec)[0].max() * 100

            if prediction == 1:
                st.error(f"🔴 Fake News Detected! ({confidence:.2f}% confidence)")
            else:
                st.success(f"🟢 Real News! ({confidence:.2f}% confidence)")

st.subheader("Step 2: Changed News Detection")
input_original_text = st.text_area("Enter Original News Text Here:", value=st.session_state.original_news, height=200)
st.session_state.original_news = input_original_text

input_changed_text = st.text_area("Enter Changed News Text Here:", height=200)

if st.button("Compare Changed News"):
    if input_original_text.strip() == "" or input_changed_text.strip() == "":
        st.error("Please enter both original and changed news text.")
    else:
        if input_original_text.strip() == input_changed_text.strip():
            st.success("✅ News is not changed.")
        elif input_changed_text.strip() in input_original_text.strip():
            st.info("ℹ️ The changed news is only a part of the original news.")
        else:
           
            try:
                from sentence_transformers import SentenceTransformer
                sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

                highlighted_text = highlight_changes(input_original_text, input_changed_text)

           
                original_emb = sentence_model.encode(input_original_text)
                changed_emb = sentence_model.encode(input_changed_text)
                similarity_score = cosine_similarity([original_emb], [changed_emb])[0][0]

                st.write(f"Similarity Score: {similarity_score:.2f}")

                if similarity_score >= 0.99 and 'style="background-color: red;"' not in highlighted_text:
                    st.success("✅ News is not changed.")
                else:
                    st.markdown(f"**Highlighted Changes:**<br>{highlighted_text}", unsafe_allow_html=True)

                    if similarity_score > 0.8:
                        st.info("The news has been altered but remains similar to the original.")
                    else:
                        st.warning("The news is significantly different from the original.")
            except Exception as e:
                st.error(f"Error using SentenceTransformer: {str(e)}")
