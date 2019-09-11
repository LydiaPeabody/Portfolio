# What can job descriptions tell us about our workforce?

The workforce in Australia is highly gender segregated, with the majority of Australians working in an [industry dominated by one gender or another](https://www.wgea.gov.au/data/fact-sheets/gender-segregation-in-australias-workforce). Why do we care? Because study after study has shown that organizations are more creative, productive, and successful with a diverse range of staff.

A small number of [well documented practices](https://business.linkedin.com/talent-solutions/blog/job-descriptions/2018/5-must-dos-for-writing-inclusive-job-descriptions) are recommended to design job descriptions that will attract candidates from multiple genders. Use of these [strategies](
https://blog.criteriacorp.com/6-simple-steps-to-reduce-gender-bias-in-your-job-descriptions) results in not just more female applicants, but a larger overall applicant field.

There are existing tools to assist organizations looking to improve in these areas. For example, the [gender decoder](http://gender-decoder.katmatfield.com/) is a free online tool that checks for use of gender-coded words. For larger-scale implementation, it's available as both an Anaconda package and a github repo, so there are few excuses not to try it out.

Don't think it matters? Have a look at the success of [predicting genders of hires based solely on analysis of job descriptions](https://textio.ai/gendered-language-in-your-job-post-predicts-the-gender-of-the-person-youll-hire-cd150452407d) at Expedia. How you advertise a role is a strong predictor of who you will hire - in many cases companies and recruiters are including more communication to potential applicants than they realize.

After reading an analysis of how gender disparity is still increasing in many Australian sectors, I decided to analyze job descriptions in an effort to understand how this piece of the recruitment process may be impacting hiring in various sectors.

In addition to looking to understand overall levels of industry compliance with recommended practices, I will be using a range of natural language processing, machine learning, and modelling techniques to understand similarities and differences in job descriptions, across industries, as well as connections to other data where possible (e.g. company ratings).

This is a work in progress - reach out if you're interested to know more, or if you have suggestions for questions or data to incorporate.

## Current Progress:
### Data Collection
Data consistes of job descriptions scraped from job search websites. I have chosen a number of industries to begin analysis, looking for a mix of industries considered gender dominated and mixed gender. These include:
* Female dominated - Nursing, Early Learning
* Mixed - Retail, Hospitality
* Male dominated - Construction, Manufacturing
Bonus industries: 
* 'Data' - as a data professional, I'm intrigued by the male dominance in a sector that has had lots of media attention.
* Finance - some evidence says finance is now a mixed industry, with no gender holding over 60% of the roles. If this is so, it should provide an interesting counterpoint to data, as both areas require similar mathematical and technical skill development.

Features generated through scraping for each job include:
Job Title, Job Description Text, Company Name, Company Rating (stars), Salary, Industry (based on search terms), Location (based on search) 

### Data Cleaning
Initial data collection resulted in about 40,000 rows of data, with each row representing one job
* Removed duplicate data - almost 50% of scraped roles are duplicates
* Dropped rows with no job description text
* Pulled numerical salary and pay period from salary text, created equivalent annual salary
* Formatted text: removing common (uninformative) words and punctuation

### EDA: Exploratory Data Analysis
I have had an initial look at the data, and begun creating additional features using NLP techniques, such as "sentiment," as well as a basic target column of industry gender dominance (mixed, female, male), based on Australian government statistics. Some features like "sentiment" show potential. Initial LDA visualization shows a clear separation between clusters appearing to represent female-dominated and male-dominated industries. Inital exploration of "gender-coded words" from prior research on job descriptions shows little connection to industry dominance. This may be a real indicator, but I suspect I need to write a better search function to handle stemmed words.

## Next Steps:
Continue feature creation and text cleaning
* Identify qualifications / requirements for jobs
* Improve cleaning processes converting JD strings to individual words
* Separate staff and "management/leadership level positions for separate analysis

Modelling
* Test both clustering and classification models relative to the industry gender domininace target
* Perform significance testing on observed differences from EDA
