//
//  CalculatorBrain.swift
//  BMI Calculator
//
//  Created by Michel Maalouli on 25/12/2021.
//

import UIKit

struct CalculatorBrain {
    var bmi: BMI?
    
    mutating func calculateBMI(height: Float, weight: Float) {
        let bmiValue = weight / pow(height, 2)
        if bmiValue < 18.5 {
            bmi = BMI(value: bmiValue, advice: "Eat more shawarma!", color: .cyan)
        } else if bmiValue < 24.9 {
            bmi = BMI(value: bmiValue, advice: "You're fit!", color: .green)
        } else {
            bmi = BMI(value: bmiValue, advice: "Eat less shawarma!!", color: .red)
        }
    }
    
    func getBMIValue() -> String {
        return String(format: "%.1f", bmi?.value ?? 0.0)
    }
    
    func getColor() -> UIColor {
        return bmi?.color ?? .clear
    }
    
    func getAdvice() -> String {
        return bmi?.advice ?? "default advice"
    }
}
