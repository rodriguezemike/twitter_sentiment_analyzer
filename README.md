# Twitter Sentiment Analyzer

**CS 175 Project: Blade Runner, A Sentiment Classifier for Twitter**

This project, originally developed for **CS 175: Project in Artificial Intelligence**, is a sentiment analysis tool designed to classify Twitter posts (tweets) as either positive or negative based on their content. The sentiment analysis is powered by two classification algorithms: **Naive Bayes** and **Logistic Regression**. Additionally, the project includes a preprocessing step for tokenizing and cleaning the tweet data before applying the classifiers.

**Note:** This project is **no longer up to date** and may not function as intended with the latest versions of libraries or APIs. It was created for educational purposes during a CS 175 course, and contributions or updates will not be accepted.

## Features

- **Naive Bayes Classifier:** A probabilistic classifier based on Bayes' Theorem, used to classify tweets as positive or negative.
- **Logistic Regression Classifier:** A linear model used for binary classification, implemented to provide an alternative to the Naive Bayes approach.
- **Token Preprocessing:** Tweets undergo preprocessing, including tokenization, to clean and prepare data for classification.
  
## Project Structure (MVC Design Pattern)

The project is structured using the **MVC (Model-View-Controller)** design pattern:

### Model:
- **Tweet Object**: Represents individual tweets with attributes such as text content, sentiment label (positive or negative), and a unique identifier.
- **Tweet Collection Object**: Contains a collection of tweets and provides methods for accessing and manipulating tweet data. It also includes methods to fetch tweets from a source (Tweet Provider).

### Controller:
- The controller layer is responsible for coordinating the interaction between the model and the view.
- **Learners**: Implements different machine learning algorithms, including Naive Bayes and Logistic Regression, to train models on tweet data and generate predictions.

### View:
- The view layer contains the classes responsible for displaying the output of the classifiers, including the sentiment labels and any additional results (e.g., accuracy, confusion matrices).

## Classifiers

- **Naive Bayes Classifier**: This classifier uses probability theory to predict the sentiment of a tweet, based on the assumption that features (words) are independent given the class (sentiment).
- **Logistic Regression Classifier**: This classifier uses a logistic function to predict the probability of a tweet being positive or negative.

Both classifiers are trained using the same tokenized tweet data, and their performance is compared.

## Contributing

This project is archived and no longer maintained. It was built as part of a course project and is not actively updated. Contributions are **not being accepted**.

If you want to use this as a starting point for a sentiment analysis project, feel free to fork and modify the repository, but please be aware that it may require updates to work with modern dependencies.

## License

This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **CS 175: Project in Artificial Intelligence** - This project was developed as part of the curriculum for the CS 175 course.
- **Scikit-learn** - A library used for machine learning algorithms and tools in this project.
- **pandas** - A library used for data manipulation and preprocessing of tweets.

---

Feel free to explore the repository and adapt the code for educational purposes or further experimentation with sentiment analysis!
