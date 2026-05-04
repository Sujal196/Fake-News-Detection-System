import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        print(f"NLTK download error: {e}")

class TextPreprocessor:
    def __init__(self):
        download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
            
        text = text.lower()
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'<.*?>', '', text)
        
        text = re.sub(r'\S+@\S+', '', text)
        
        text = re.sub(r'\d{3}[-.]?\d{3}[-.]?\d{4}', '', text)
        
        text = re.sub(r'[^\w\s\.\!\?\,\:\;]', '', text)
        
        text = re.sub(r'\b(?!(?:19|20)\d{2}\b)\d+\b', '', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_remove_stopwords(self, text):
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens
    
    def lemmatize_tokens(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text):
        cleaned_text = self.clean_text(text)
        
        tokens = self.tokenize_and_remove_stopwords(cleaned_text)
        
        lemmatized_tokens = self.lemmatize_tokens(tokens)
        
        processed_text = ' '.join(lemmatized_tokens)
        
        return processed_text

def load_and_preprocess_data(file_path=None):
    if file_path and pd.io.common.file_exists(file_path):
        df = pd.read_csv(file_path)
    else:
        data = {
            'text': [
                "Scientists at Stanford University have published a groundbreaking study in Nature Medicine showing promising results for a new cancer treatment approach. The research, led by Dr. Sarah Johnson, analyzed data from over 500 patients with advanced melanoma and found that the experimental therapy reduced tumor size by an average of 45%. The treatment combines immunotherapy with targeted drug delivery, showing fewer side effects compared to traditional chemotherapy. Clinical trials began in 2021 and are expected to continue through 2024, with early results indicating significant improvements in patient survival rates. The research was funded by the National Institutes of Health and several private foundations dedicated to cancer research.",
                
                "Researchers at MIT have developed a new artificial intelligence algorithm that can predict protein structures with unprecedented accuracy. The breakthrough, published in Science journal, could revolutionize drug discovery and disease treatment. The algorithm, called AlphaFold-Pro, uses deep learning techniques to analyze amino acid sequences and predict how proteins will fold in three-dimensional space. This capability could significantly reduce the time and cost required to develop new medications. The research team tested the algorithm on thousands of known protein structures and achieved 92% accuracy in predicting their configurations. Major pharmaceutical companies have already expressed interest in licensing the technology for drug development purposes.",
                
                "The Federal Reserve announced today that it will maintain current interest rates following a comprehensive review of economic indicators. Chairman Jerome Powell stated that inflation has shown signs of moderation but remains above the target range of 2%. The decision comes after the central bank raised rates by 0.25% at their previous meeting in an effort to curb rising prices. Economic data released this week showed that consumer prices increased by 0.3% in July, bringing the annual inflation rate to 3.2%. The unemployment rate remains steady at 3.8%, with job growth continuing in sectors such as healthcare, technology, and construction. Financial markets reacted positively to the announcement, with major stock indices rising by more than 1% in afternoon trading.",
                
                "Congress passed bipartisan legislation today aimed at strengthening cybersecurity protections for critical infrastructure across the United States. The bill, which received overwhelming support with a 412-23 vote in the House and 87-13 in the Senate, establishes new standards for power grids, water systems, and transportation networks. The legislation includes $50 billion in funding for state and local governments to upgrade their cybersecurity systems and creates a federal office to coordinate responses to major cyber attacks. The bill comes in response to increasing threats from foreign adversaries and criminal organizations targeting essential services. President Biden is expected to sign the legislation into law tomorrow at a ceremony at the White House.",
                
                "Apple Inc. reported quarterly earnings that exceeded analyst expectations, driven by strong sales of iPhones and services. The tech giant posted revenue of $89.5 billion for the fiscal third quarter, up 8% from the same period last year. iPhone sales reached $39.7 billion, while the services division grew 12% to $21.2 billion. CEO Tim Cook attributed the strong performance to robust demand for the iPhone 14 series and continued growth in the App Store, Apple Music, and iCloud services. The company also announced that it has invested over $20 billion in U.S. operations this year, including new data centers and manufacturing facilities. Apple's stock price rose by 4% in after-hours trading following the earnings announcement.",
                
                "Microsoft announced a major expansion of its cloud computing infrastructure with the opening of three new data centers in Europe. The $2 billion investment will bring Microsoft Azure services to more regions and improve performance for enterprise customers. The new data centers, located in Germany, Spain, and Poland, will be powered entirely by renewable energy as part of Microsoft's commitment to carbon neutrality by 2030. The expansion comes as demand for cloud services continues to grow, with Microsoft reporting 27% year-over-year growth in its Azure cloud platform. The company also announced new partnerships with European automotive and manufacturing companies to migrate their operations to Microsoft's cloud infrastructure.",
                
                "The World Health Organization has approved a new malaria vaccine that could save hundreds of thousands of lives annually in Africa. The vaccine, developed by GlaxoSmithKline and tested in clinical trials involving over 800,000 children, prevents about 40% of malaria cases and 30% of severe cases. Dr. Tedros Adhanom Ghebreyesus, WHO Director-General, called the approval a historic moment in the fight against malaria. The vaccine will be rolled out initially in Ghana, Kenya, and Malawi, with plans to expand to other high-risk countries. The development of the vaccine took over 30 years and cost more than $700 million, with funding from governments, foundations, and pharmaceutical companies.",
                
                "A comprehensive study published in The Lancet has found that regular physical activity can reduce the risk of developing dementia by up to 30%. The research, which followed 200,000 participants over 20 years, is the largest study to date examining the relationship between exercise and cognitive decline. Participants who engaged in at least 150 minutes of moderate exercise per week showed significantly better cognitive function and lower rates of Alzheimer's disease and other forms of dementia. The study's authors recommend that adults incorporate aerobic exercise, strength training, and balance exercises into their weekly routines to maintain brain health. The findings have important implications for public health policy as the global population ages and dementia rates continue to rise.",
                "The European Union has implemented ambitious new climate legislation aimed at reducing carbon emissions by 55% by 2030. The package of measures, known as 'Fit for 55', includes stricter emissions standards for vehicles, increased renewable energy targets, and a carbon border tax on imports from countries with weaker environmental regulations. The legislation represents the most significant climate action taken by any major economic bloc to date and could serve as a model for other regions. EU officials estimate that the measures will create millions of green jobs and reduce energy imports by 40% over the next decade. The plan also includes funding for developing countries to transition to cleaner energy sources and adapt to climate change impacts.","Scientists at the National Oceanic and Atmospheric Administration have confirmed that 2023 was the hottest year on record, with global temperatures averaging 1.2 degrees Celsius above pre-industrial levels. The annual climate report, which analyzes data from thousands of weather stations and ocean buoys worldwide, shows that all seven of the hottest years on record have occurred since 2015. Arctic sea ice reached its second-lowest extent on record, while ocean temperatures continue to rise at unprecedented rates. The findings underscore the urgent need for action to reduce greenhouse gas emissions and transition to renewable energy sources. Climate scientists warn that without significant reductions in emissions, global temperatures could rise by more than 3 degrees Celsius by the end of the century.",
                
                "Harvard University has announced the largest endowment gift in its history, a $500 million donation from an anonymous alumnus to establish a new institute for artificial intelligence research. The gift will fund faculty positions, graduate scholarships, and research facilities at the Harvard John A. Paulson School of Engineering and Applied Sciences. The institute will focus on developing ethical AI systems and studying the societal impacts of artificial intelligence. University officials said the donation will enable Harvard to become a global leader in AI research and education. The anonymous donor, who graduated from Harvard in the 1980s, has previously supported other educational initiatives but has never before made a gift of this magnitude.",
                
                "A landmark study by the Brookings Institution has found that early childhood education programs provide significant long-term economic benefits to society. The research, which analyzed data from 50 years of educational programs, shows that every dollar invested in high-quality preschool education generates $7 in economic returns through increased earnings, reduced crime rates, and lower healthcare costs. Children who participated in quality early education programs were more likely to graduate from high school, attend college, and secure stable employment. The study's authors recommend that federal and state governments increase funding for universal preschool programs, particularly for low-income families who benefit most from early educational interventions.",
                
                                "BREAKING: Doctors are furious about this one simple trick that reverses aging overnight! A 78-year-old grandmother from Florida discovered a secret combination of common household ingredients that can make anyone look 20 years younger in just 24 hours. Big pharmaceutical companies are spending millions to keep this information from the public because it would destroy their anti-aging industry worth billions. The formula includes coconut oil, turmeric, and a rare herb found only in the Amazon rainforest. People who have tried it report wrinkles disappearing, gray hair turning back to its original color, and energy levels they haven't felt since they were teenagers. This is the real fountain of youth that the medical establishment doesn't want you to know about.",
                
                "SHOCKING DISCOVERY: Miracle plant from the Amazon jungle cures all forms of cancer in just 48 hours! Indigenous healers have known about this plant for centuries but Western medicine has been suppressing the information to protect their lucrative cancer treatment industry. The plant, which cannot be cultivated outside its natural habitat, contains compounds that target and destroy cancer cells while leaving healthy cells completely untouched. A team of independent researchers who smuggled samples out of Brazil have documented complete remission in 97% of terminal cancer patients who took the extract. The pharmaceutical industry has allegedly bribed government officials to classify the plant as a controlled substance to prevent public access. This is the biggest medical conspiracy in human history.",
                
                "ALIEN SPACECRAFT FOUND: Government confirms UFO crash site in Nevada desert but covering up the truth! Multiple whistleblowers from Area 51 have revealed that a spacecraft from another galaxy crashed in 2019 and the government recovered three alien bodies. The aliens, described as being 4 feet tall with large heads and telepathic abilities, were taken to a secret underground facility where they communicated with scientists. They warned humanity about an impending invasion from their home planet scheduled for 2025. The government is hiding this information to prevent mass panic while secretly building underground bunkers for the elite. Satellite images show unusual activity at the crash site, and nearby residents have reported strange lights and unexplained phenomena for months.",
                
                "TIME TRAVELERS FROM 2050 WARNING: Future humans reveal shocking truth about what's coming! A group of time travelers who arrived from the year 2050 have been secretly meeting with world leaders to warn them about catastrophic events that will devastate Earth in the coming decades. They claim that climate change will become irreversible by 2030, leading to mass extinctions and the collapse of civilization. The time travelers, who appear as normal humans but possess advanced technology, have provided detailed predictions that have already begun to come true. They say they traveled back to prevent the worst outcomes but are being blocked by powerful corporations who profit from the current system. Several world leaders have resigned after receiving these warnings, but the media is covering up the truth.",
                
                "EXCLUSIVE: Top politician caught in shocking scandal that will bring down the entire government! Multiple sources have confirmed that a senior cabinet minister has been operating a secret offshore bank account containing over $100 million in bribes from foreign governments. The minister, who cannot be named due to ongoing investigations, allegedly sold state secrets and awarded government contracts to companies in exchange for kickbacks. The scandal involves officials at the highest levels of government and could lead to dozens of arrests and resignations. Evidence includes bank records, emails, and testimony from former associates who have turned whistleblower. The mainstream media is refusing to report on this story due to pressure from government officials who are threatening legal action against anyone who publishes the information.",
                
                "SECRET GOVERNMENT PROGRAM EXPOSED: Mind control technology being used on millions of citizens without their knowledge! Leaked documents from a classified military program reveal that government agencies have been testing advanced mind control technology on unsuspecting Americans for over a decade. The program, called 'Project Silent Voice,' uses subliminal messages embedded in television broadcasts, social media, and even popular music to influence public opinion and behavior. Former military psychologists who worked on the project claim the technology can make people support specific policies, buy certain products, or even vote for particular candidates. The whistleblowers have gone into hiding after receiving death threats, but they've provided extensive documentation including patents, internal memos, and test results that prove the program exists.",
                
                "MIRACLE INVESTMENT: Secret cryptocurrency discovered that will make ordinary investors millionaires by next month! A team of anonymous hackers has uncovered a little-known cryptocurrency that is set to explode in value by over 10,000% in the coming weeks. The coin, which currently trades for just $0.01, has been secretly developed by a group of Silicon Valley billionaires who plan to use it to replace traditional banking systems. Insiders at major tech companies have been quietly accumulating the cryptocurrency before its public launch. Financial experts who have analyzed the technology say it could become the world's most valuable digital asset within months. This is a once-in-a-lifetime opportunity to get in early before the price skyrockets. Regular people who invest even $100 could become millionaires by the end of the year.",
                
                "ANCIENT TREASURE FOUND: Archaeologists discover gold worth $50 billion that could change the global economy! A team exploring a newly discovered tomb in Egypt has uncovered the largest collection of gold and precious gems ever found in history. The treasure, which includes solid gold statues, jewel-encrusted artifacts, and rare coins, is estimated to be worth over $50 billion at current market prices. However, the Egyptian government is trying to keep the discovery secret to avoid disrupting world financial markets. The artifacts date back over 3,000 years and include items that could rewrite human history. Several archaeologists involved in the discovery have mysteriously disappeared, and the excavation site has been closed to the public. Some experts believe the treasure contains evidence of advanced ancient technology that modern scientists cannot explain.",
                
                "MIRACLE SIGN: End times prophecy fulfilled as mysterious lights appear in the sky worldwide! Millions of people have reported seeing strange lights and formations in the sky that match ancient prophecies about the end of the world. Religious scholars who have studied biblical texts say these signs indicate that the apocalypse will begin within the next six months. The lights have been documented by astronomers who cannot explain their origin or behavior. Similar phenomena were reported before major historical events, including the fall of Rome and World War II. Religious leaders are urging people to prepare for judgment day, while scientists are baffled by the unexplained phenomena. Military radar has detected the objects performing maneuvers that defy known laws of physics, suggesting they are not natural occurrences.",
                
                "ANGEL APPEARANCE: Real angels caught on camera performing miracles in hospital! Multiple security cameras at a major hospital captured what appears to be actual angels visiting patients and performing miraculous healings. In one video, a winged figure is seen touching a terminally ill patient who immediately recovers from her illness. In another, doctors and nurses watch in amazement as a glowing being enters the operating room and guides surgeons during a complicated procedure. The hospital administration has tried to suppress the videos, but staff members have leaked them to religious organizations. Medical experts who have examined the footage cannot explain how the patients recovered so quickly or what the winged figures could be. Religious leaders are calling this definitive proof of divine intervention in human affairs.",
                
                "BREAKTHROUGH TECHNOLOGY: Scientists invent free energy device that could end all energy costs forever! A team of independent researchers has created a revolutionary device that generates unlimited electricity from zero-point energy, completely eliminating the need for fossil fuels, nuclear power, or renewable energy sources. The device, which is smaller than a shoebox and costs less than $100 to build, can power an entire home indefinitely without any fuel or maintenance. Major energy companies have allegedly threatened the researchers and tried to suppress the technology to protect their trillion-dollar profits. However, the scientists have released detailed plans online so anyone can build their own device. Early adopters report saving thousands of dollars on electricity bills and some have even started selling excess power back to utility companies.",
                
                "QUANTUM COMPUTER BREAKTHROUGH: New machine can predict the future with 100% accuracy! Researchers at a secret laboratory have developed a quantum computer that can accurately predict future events, including stock market movements, natural disasters, and even personal life events. The machine uses advanced quantum mechanics to analyze all possible future scenarios and determine which one will actually occur. Initial tests have shown 100% accuracy in predicting everything from lottery numbers to election results. The government has classified the technology as a national security threat because it could be used for terrorism or market manipulation. However, the scientists who developed it believe it should be used to prevent disasters and help humanity prepare for future challenges. Some early predictions have already come true, including several major earthquakes and political events."
            ],
            'label': [1] * 12 + [0] * 12  # 12 Real News, 12 Fake News
        }
        df = pd.DataFrame(data)
    
    preprocessor = TextPreprocessor()
    
    df['processed_text'] = df['text'].apply(preprocessor.preprocess_text)
    
    return df

def extract_features_tfidf(texts, max_features=10000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        stop_words='english'
    )
    features = vectorizer.fit_transform(texts)
    return features, vectorizer

def prepare_data(df):
    X = df['processed_text']
    y = df['label']
    
    X_tfidf, tfidf_vectorizer = extract_features_tfidf(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, tfidf_vectorizer

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Sample processed text: {df['processed_text'].iloc[0]}")
    
    X_train, X_test, y_train, y_test, vectorizer = prepare_data(df)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
