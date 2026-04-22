import nltk
from nltk.chat.util import Chat, reflections
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('stopwords')
nltk.download('punkt_tab')

response_pairs = [
    (r"hi|hello|hey", ["Hello!", "Hi there!", "Hey!"]),
    (r"my name is (.*)", ["Nice to meet you, %1!"]),
    (r"how are you", ["I'm doing well, thank you!", "I'm great, how about you?"]),
    (r"i am (good|well|okay|ok)", ["That's good to hear!"]),
    (r"what do you do", ["I'm a chatbot, I can help answer questions or have a conversation."]),
    (r"quit", ["Goodbye! Have a great day!"]),
    (r"When does the exam season start and end?", ["The exam season starts at the end of the semester but course work can be due at any time."]),
    (r"How can I access my exam timetable? ", ["You can access your timetable on Beacon"]),
    (r"Are there any changes in exam locations due to COVID-19? ", ["Exams were affected, they were held online or hybrid."]),
    (r"What materials are allowed during exams? ", ["Usually there are no materials allowed in the exam but some assessments allow open book."]),
    (r"How do I request special accommodations for my exams? ", ["You can speak to student support to discuss accommodations."]),
    (r"What is the university's policy on academic integrity during exams? ", ["Submit only your own work with proper referecning."]),
    (r"Are calculators allowed during exams, and if so, which models? ", ["Calculators are allowed in some assessments, there many models that are allowd the common ones are Casio FX-83GT and FX-991CW."]),
    (r"What should I do if I experience technical issues during an online exam? ", ["Notify your module leader and the assessment lead quickly."]),
    (r"What is the process for appealing an exam grade? ", ["The prosses consists of submitting an appeal on e:Vision within 2 weeks of your grade release."]),
    (r"Are there any study resources available for students during the exam season? ", ["24/7 library access, academic skills appointments, study spaces, extended labs access and one to one support."]),
    (r"Is there a deadline to withdraw from a course during the exam period? ", ["There is no deadline however it is not reccommended as it may cause issues in the future."]),
    (r"Can I reschedule an exam if I have a scheduling conflict or medical emergency? ", ["Yes, you can reschedule if you have valid reasons like a medical emergency."]),
    (r"What should I do if I arrive late to an exam? ", ["Depending on the module and the lecturer you may have to wait till a different slot, have to resit at a later time or try to complete the exam in the time remaining."]),
    (r"Are there any restrictions on personal items in the exam room?", ["Yes, smart devices and notes are banned."]),
    (r"What is the policy on bathroom breaks during exams? ", ["You are not allow to leave the room without the permition of an invigilator and cannot go in the first hour or last 30 mins of the exam."]),
    (r"Can I bring a (snack*, food*) or drink into the exam room? ", ["Yes, small snacks and bottled drinks are allowed."]),
    (r"How long are the exam periods and are there breaks between exams? ", ["Exam periods are usually 2-3 weeks with one exam per day."]),
    (r"Are there designated quiet study areas on campus during exam season? ", ["Yes, the university has 2 libraries and designated study areas in the Catalyst, Mellor and Cadman Buildings."]),
    (r"Are there any workshops or tutoring services available to help with exam preparation? ", ["Yes but it will depend on the module and course."]),
    (r"How can I manage exam stress and anxiety? ", ["You, can book a wellbeing appointment to discuss your mental health at anytime."]),
    (r"What should I do if I feel unwell during an exam? ", ["Before the exam let the lecturer know to try and reschedule if you do during the exam, ask an invigilator and leave the room till you feel well enough to continue the exam."]),
    (r"How and when will I receive my exam results? ", ["You will recive your mark on blackboard with the official results being released on e:Vision. Lecturers have upto 20 working days to mark and submit the grades."]),
    (r"Can I request a review of my exam paper? ", ["Yes, but only if you have evidence of miscalculation of your grade."]),
    (r"Are there any penalties for missing an exam without prior notice? ", ["Yes, missing an exam without prior notice will result in a score of 0 and having to resit the exam."]),
    (r"Are students required to wear face masks during in-person exams? ", ["During Covid-19 yes however now masks are not required."]),
    (r"Is there a minimum passing grade for exams? ", ["The passing grade is 40% of the total marks for your module."]),
    (r"How are exams weighted in my overall course grade? ", ["Second Year (Level 5) is weigted at 30% of your grade and Third Year (Level 6) is weighted at 70%."]),
    (r"Are there any resources for dealing with test anxiety? ", ["Yes, you can book a wellbeing appointment at any point. They can be face to face or online."]),
    (r"What do I need to bring with me to my exams? ", ["Depending on the course you might need a various items. Some could include Pen and Calculator"]),
    (r"How can I find information on exam policies specific to my department?", ["You can find exam related policies on black board under the exams banner."])
]


class Chatbot(Chat):
    def __init__(self, pairs, reflections):
        super().__init__(pairs, reflections)
        self.stemmed_token_pairs = []
        
        for question, response in pairs:
            pair = (Chatbot.languageProcessing(question), response)
            self.stemmed_token_pairs.append(pair)
        
    def similarityScore(self, userinput):
        userinput = self.languageProcessing(userinput)
        best_score = 0
        best_response = "Sorry, can you rephrase that?"
        for question, response in self.stemmed_token_pairs:
            score = len(userinput.intersection(question)) / len(userinput.union(question))
            if score > best_score:
                best_score = score
                best_response = response
            print(f"question: {question}, score: {score}, response: {response}")
        if best_score < 0.1:
            best_response = "Sorry, can you rephrase that?"
        print(f"Best Score: {best_score}, Best Response: {best_response}")
        return best_response
    
    @staticmethod
    def languageProcessing(text):
        stemmer = PorterStemmer()
        text = text.lower()
        tokenized = nltk.word_tokenize(text)
        alnum_tokenized = []

        for token in tokenized:
            if token.isalnum():
                alnum_tokenized.append(token)

        stopword_tokenized = []
        for word in alnum_tokenized:
            if word not in stopwords.words("english"):
                stopword_tokenized.append(word)

        stemmed = []
        for word in stopword_tokenized:
            stemmed.append(stemmer.stem(word))

        return set(stemmed)


def main():
    my_chatbot = Chatbot(response_pairs, reflections)
    print("Hello! I'm your chatbot, Type quit to exit.")

    while True:
        user_input = input("> ")
        if user_input.lower() == "quit":
            break

        responce = my_chatbot.respond(user_input)
        if responce is None:
            responce = my_chatbot.similarityScore(user_input)
        print(responce)

if __name__ == "__main__":
    main()