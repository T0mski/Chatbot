#Install NLTK
# Type 'pip install nltk' in the terminal

# Import nltk libraries
import nltk
from nltk.chat.util import Chat, reflections
from nltk.stem import PorterStemmer

# Download nltk tools
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# NLP components and NLP table
def perform_nlp_analysis(text):
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        stemmer = PorterStemmer()

        # print("-" * 40)
        # print(f"{'Token':<15} | {'POS Tag':<10} | {'Stem':<10}")
        # print("-" * 40)
        for word, tag in pos_tags:
            stem = stemmer.stem(word)
            # print(f"{word:<15} | {tag:<10} | {stem:<10}")
        # print("-" * 40)

# Generic responses
response_pairs1 = [
    (r"Hi|Hello|Hey", ["Hello!", "Hi there!", "Hey!"]),
    (r"My name is (.*)", ["Nice to meet you, %"]),
    (r"How are you", ["I'm doing well, thank you!", "I'm great, how about you?"]),
    (r"I am | Im (good|well|okay|ok)", ["That's good to hear!"]),
    (r"What do you do", ["Im a chatbot, I can help answer questions relating to the University of Staffordshire."]),
    (r"Quit", ["Goodbye! Have a great day!"])
]

# University specific responses
response_pairs2 = [
    (r"When does the exam season start and end?", ["Semester 1: Monday 12th January - Friday 23rd January, Semester 2: Monday 11th May - Thursday 21st May."]),
    (r"How can i access my exam timetable?", ["Your timetable can be accessed via the Beacon app or the Beacon website, you will need your University email and password to log in."]),
    (r"Are there any changes in exam locations due to COVID-19?", ["As of 2026 there are no active COVID-19 restrictions affecting exam locations."]),
    (r"What materials are allowed during exams?", ["You must bring your University ID card into the exam and basic stationary or calculators can be used if said by the module leader."]),
    (r"How do I request special accommodations for my exams?", ["To request special accommodation you must contact the inclusion team to create a student inclusion plan, this must be done at least 6 weeks before the exam period."]),
    (r"What is the university's policy on academic integrity during exams?", ["The University has zero tolerance for academic misconduct, this includes plagiarism, collusion and using unauthorized devices during exams."]),
    (r"Are calculators allowed during exams, and if so, which models?", ["Calculators are only allowed in exams if specified by your module , programmable calculators are not allowed in exams as they can be used to cheat."]),
    (r"What should I do if I experience technical issues during an online exam?", ["Take a screenshot or evidence of the error and inform your module leader and the digital services helpdesk."]),
    (r"What is the process for appealing an exam grade?", ["You can submit an academic appeal via e:Vision within 10 working days of you receiving your result."]),
    (r"Are there any study resources available for students during the exam season?", ["The academic skills team provides workshops on revision and exam techniques, the library also offers subject guides."]),
    (r"Is there a deadline to withdraw from a course during the exam period?", ["Withdrawal is handled by your student registry and you should consult a student advisor before the exam period begins."]),
    (r"Can I reschedule an exam if I have a scheduling conflict or medical emergency?", ["You cannot reschedule for personal convenience however you can for medical emergencies by filing an exceptional circumstances claim."]),
    (r"What should I do if I arrive late to an exam?", ["If you arrive late to an exam, immediately report to the invigilator. You won't be allowed into the exam if you are 30 minutes late and you wont get extra time."]),
    (r"Are there any restrictions on personal items in the exam room?", ["Yes watches, mobile phones or smart devices must be turned off and placed away from you in exam rooms."]),
    (r"What is the policy on bathroom breaks during exams?", ["You may use the bathroom during an exam however you must raise your hand and be escorted to the bathroom by an invigilator."]),
    (r"Can I bring a snack or drink into the exam room?", ["You are allowed to bring a clear bottle of water into the exam however snacks are generally not allowed unless you have a medical condition"]),
    (r"How long are the exam periods and are there breaks between exams?", ["Examinations usually last between 1 - 3 hours and if you have more than one exam in a single day the University ensures you have a gap between them."]),
    (r"Are there designated quiet study areas on campus during exam season?", ["The library, catalyst building, science center and Cadman building all have dedicated study areas."]),
    (r"Are there any workshops or tutoring services available to help with exam preparation?", ["Prior to examinations, departments offer study support both in person and online."]),
    (r"How can I manage exam stress and anxiety?", ["The student wellbeing team offers one to one support both in person and online."]),
    (r"What should I do if I feel unwell during an exam?", ["If you feel ill during an exam, raise your hand and inform an invigilator. If you are too ill to sit the exam you must submit an exceptional circumstances claim via e:Vision."]),
    (r"How and when will I receive my exam results?", ["Exam results are released on Blackboard typically 4-5 weeks after the exam period ends."]),
    (r"Can I request a review of my exam paper?", ["You can request to see your feedback and you can also request a clerical check to ensure all marks were added up correctly."]),
    (r"Are there any penalties for missing an exam without prior notice?", ["Yes you will be marked absent and your score will be 0%."]),
    (r"Are students required to wear face masks during in-person exams?", ["As of 2026 face masks are not mandatory during in person examinations."]),
    (r"Is there a minimum passing grade for exams?", ["For undergraduate modules the passing grade is 40% whilst postgraduate is 50%."]),
    (r"How are exams weighted in my overall course grade?", ["Weighting varies by module, some are 100% exam based while others may be 50/50 with coursework and assignments."]),
    (r"Are there any resources for dealing with test anxiety?", ["The wellbeing team offer help to anyone struggling with anxiety and you can get in contact and book an appointment online."]),
    (r"What do I need to bring with me to my exams?", ["You must bring your student ID card so invigilators can confirm your identity."]),
    (r"How can I find information on exam policies specific to my department?", ["Consult the module details on Blackboard or contact your course leader for more information."]),
]

# Chatbot code
stemmer = PorterStemmer()

# Preprocess function
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    return {stemmer.stem(w) for w in tokens if w.isalnum()}

class MyChatbot(Chat):
    def __init__(self, pairs, reflections):
        self.processed_pairs = []
        super().__init__(pairs, reflections)
        
        for pattern, response in pairs:
            self.processed_pairs.append({
                'stems': preprocess(pattern),
                'response' : response[0] if isinstance(response, list) else response
            })

    # Respond function
    def respond(self, str):
        regex_response = super().respond(str)
        if regex_response:
            return regex_response
        
        user_stems = preprocess(str)
        if not user_stems:
            return "Im sorry, please enter your request again."
        
        best_match = None
        highest_score = 0

        for pair in self.processed_pairs:
            match_count = len(user_stems.intersection(pair['stems']))
            score = match_count / len(user_stems)

            if score > highest_score:
                highest_score = score
                best_match = pair['response']

        if highest_score > 0.4:
            return best_match
    
        return "Im sorry, please enter your request again."

def main():
    all_pairs = response_pairs1 + response_pairs2
    my_chatbot = MyChatbot(all_pairs, reflections)

    #Homepage displays when program is run
    print()
    print("=" * 60)
    print("About".center(60))
    print("=" * 60)
    print("I am a rule-based chatbot built using the NLTK library. ")
    print()
    print("I am designed to answer questions regarding the University") 
    print("of Staffordshire's exam policies and student support tools.")
    print()
    print("I can also engage in simple conversation, try saying Hello!")
    print("=" * 60)
    print("Commands".center(60))
    print("=" * 60)
    print("'list' - View all questions you can ask me")
    print("'search - Search for questions by keyword")
    print("'quit' - Exit the program")
    print("=" * 60)
    print()
    print("How can I help you today?")
    print()
    
    while True:
        user_input = input("User: ")
        print()

        # NLP analysis 
        perform_nlp_analysis(user_input)

        # New 'list' command to print available requests
        if user_input.lower() == 'list':
            print()
            print("--- Available Requests ---")
            for question, response in response_pairs2:
                print(f"- {question}")
            print()
            continue

        # 'quit' command to end program
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        # New 'search' command to find questions containing a keyword
        if user_input.lower().startswith('search'):
            keyword = user_input.lower().replace('search', '').strip()
            if not keyword:
                print("Please enter 'search' and your keyword, e.g. 'search exam'")
                print()
                continue
    
            results = [q for q, r in response_pairs2 if keyword in q.lower()]
    
            if results:
                print(f"--- Questions containing '{keyword}' ---")
                for question in results:
                    print(f"- {question}")
                print()
                
            else:
                print(f"No questions found containing '{keyword}'.")
                print()
            continue
        
        response = my_chatbot.respond(user_input)
        print(f"Bot: {response}")
        print()

if __name__ == "__main__":
    main()