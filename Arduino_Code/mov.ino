#include <Servo.h>

Servo myservo;
const int pwmPin = 5;
const int motorPin = 11;
const int pwmSpeed = 160;
const int servoStep = 5; // Define the step size for servo movement
const int servoDelay = 100; // Adjust the delay for smoother movement

unsigned long forwardStartTime = 0;
unsigned long stopStartTime = 0;

void setup() {
  Serial.begin(9600);
  pinMode(pwmPin, OUTPUT);
  pinMode(motorPin, OUTPUT);
  myservo.attach(9); // Attach the servo to pin 9
}

void loop() {
  if (Serial.available() > 0) {
    char receivedChar = Serial.read();
    if ((receivedChar >= 'a') && (receivedChar <= 'g')) {
      int targetAngle = map(receivedChar, 'a', 'g', 50, 130);

      // Ensure the mapped angle is within the valid range
      targetAngle = constrain(targetAngle, 50, 130);

      // Gradual servo movement
      smoothServoMove(targetAngle);
      forward();
    }
    else if (receivedChar == 's') {
      stop();
    }
  }
}

void smoothServoMove(int targetAngle) {
  int currentAngle = myservo.read();
  unsigned long previousMillis = 0;

  while (currentAngle != targetAngle) {
    unsigned long currentMillis = millis();

    if (currentMillis - previousMillis >= servoDelay) {
      previousMillis = currentMillis;

      if (currentAngle < targetAngle) {
        currentAngle += servoStep;
        currentAngle = min(currentAngle, targetAngle);
      } else {
        currentAngle -= servoStep;
        currentAngle = max(currentAngle, targetAngle);
      }

      // Move the servo to the updated angle
      myservo.write(currentAngle);
    }
  }
}

void forward() {
  digitalWrite(motorPin, HIGH);
  analogWrite(pwmPin, pwmSpeed);

  // Store the start time for forward movement
  if (forwardStartTime == 0) {
    forwardStartTime = millis();
  }

  // Add a delay equivalent using millis()
  if (millis() - forwardStartTime >= 10) {
    forwardStartTime = 0; // Reset the start time
  }
}

void stop() {
  digitalWrite(motorPin, LOW);
  digitalWrite(pwmPin, LOW);
  delay(10);
}