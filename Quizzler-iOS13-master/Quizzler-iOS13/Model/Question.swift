//
//  Question.swift
//  Quizzler-iOS13
//
//  Created by Michel Maalouli on 23/12/2021.
//

import Foundation

struct Question {
    let text: String
    let choices: [String]
    let answer: String
    
    init(q: String, a: [String], correctAnswer: String) {
        text = q
        choices = a
        answer = correctAnswer
    }
}
