//
//  ViewController.swift
//  Tipsy
//
//  Created by Michel Maalouli on 28/12/2021.
//

import UIKit

class CalculatorViewController: UIViewController {
    @IBOutlet weak var billTextField: UITextField!
    @IBOutlet weak var zeroPctButton: UIButton!
    @IBOutlet weak var tenPctButton: UIButton!
    @IBOutlet weak var twentyPctButton: UIButton!
    @IBOutlet weak var splitNumberLabel: UILabel!
    
    var tip = 0.0
    var percentTip = ""
    var result = ""

    @IBAction func tipChanged(_ sender: UIButton) {
        billTextField.endEditing(true)
        if sender.currentTitle == "0%" {
            zeroPctButton.isSelected = true
            tenPctButton.isSelected = false
            twentyPctButton.isSelected = false
        } else if sender.currentTitle == "10%" {
            zeroPctButton.isSelected = false
            tenPctButton.isSelected = true
            twentyPctButton.isSelected = false
        } else if sender.currentTitle == "20%" {
            zeroPctButton.isSelected = false
            tenPctButton.isSelected = false
            twentyPctButton.isSelected = true
        }
    }
    
    @IBAction func stepperValueChanged(_ sender: UIStepper) {
        billTextField.endEditing(true)
        splitNumberLabel.text = Int(sender.value).description
    }
    
    @IBAction func calculatePressed(_ sender: UIButton) {
        if zeroPctButton.isSelected {
            tip = 0.0
            percentTip = zeroPctButton.currentTitle!
        } else if tenPctButton.isSelected {
            tip = 0.1
            percentTip = tenPctButton.currentTitle!
        } else if twentyPctButton.isSelected {
            tip = 0.2
            percentTip = twentyPctButton.currentTitle!
        }
        if billTextField.text == "" {
            let alert = UIAlertController(title: "Invalid Bill Total", message: "Please provide valid input for bill.", preferredStyle: UIAlertController.Style.alert)

            // add an action (button)
            alert.addAction(UIAlertAction(title: "OK", style: UIAlertAction.Style.default, handler: nil))

            // show the alert
            self.present(alert, animated: true, completion: nil)
        } else {
            print(tip)
            print(splitNumberLabel.text!)
            print(billTextField.text!)
            result = String(format: "%.2f", (1.0 + Float(tip)) * Float(billTextField.text!)! / Float(splitNumberLabel.text!)!)
        }
        
        performSegue(withIdentifier: "goToResult", sender: self)
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        let destinationVC = segue.destination as! ResultsViewController
        destinationVC.result = result
        destinationVC.numberOfPeople = splitNumberLabel.text
        destinationVC.tipPercentage = percentTip
    }
}

