//
//  ViewController.swift
//  Quizzler-iOS13
//
//  Created by Michel Maalouli on 12/25/2021.
//

import UIKit

class ViewController: UIViewController {
    
    @IBOutlet weak var scoreLabel: UILabel!
    @IBOutlet weak var questionLabel: UILabel!
    @IBOutlet weak var progressBar: UIProgressView!
    @IBOutlet weak var ans1Button: UIButton!
    @IBOutlet weak var ans2Button: UIButton!
    @IBOutlet weak var ans3Button: UIButton!
    
    var quizBrain = QuizBrain()
 
    override func viewDidLoad() {
        super.viewDidLoad()
        updateUI()
    }

    @IBAction func answerButtonPressed(_ sender: UIButton) {
        if quizBrain.checkAnswer(sender.currentTitle!) {
            sender.backgroundColor = UIColor.green
        } else {
            sender.backgroundColor = UIColor.red
        }
        
        quizBrain.nextQuestion()
        
        Timer.scheduledTimer(timeInterval: 0.2, target: self, selector: #selector(updateUI), userInfo: nil, repeats: false)
    }
    
    @objc func updateUI() {
        questionLabel.text = quizBrain.getQuestionText()
        ans1Button.setTitle(quizBrain.getAnswers()[0], for: .normal)
        ans2Button.setTitle(quizBrain.getAnswers()[1], for: .normal)
        ans3Button.setTitle(quizBrain.getAnswers()[2], for: .normal)
        progressBar.progress = quizBrain.getProgress()
        scoreLabel.text = "Score: \(quizBrain.getScore())"
        ans1Button.backgroundColor = UIColor.clear
        ans2Button.backgroundColor = UIColor.clear
        ans3Button.backgroundColor = UIColor.clear
    }
    
}

