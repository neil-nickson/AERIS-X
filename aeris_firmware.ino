// ==========================================
// AERIS-X PHASE 1 FIRMWARE
// ==========================================
//
// WIRING:
// DHT11 DATA       -> D2
// Rain sensor DO   -> D3
// LDR AO           -> A0
// ADXL335 X        -> A1
// ADXL335 Y        -> A2
// ADXL335 Z        -> A3
// Sensor VCC       -> 5V
// Sensor GND       -> GND
// ==========================================

#include <DHT.h>

#define DHTPIN 2
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

#define RAIN_PIN 3
#define LDR_PIN A0
#define ADXL_X A1
#define ADXL_Y A2
#define ADXL_Z A3

void setup() {
  Serial.begin(9600);

  pinMode(RAIN_PIN, INPUT);
  // LDR and ADXL are analog inputs, pinMode is optional but good practice
  pinMode(LDR_PIN, INPUT);
  pinMode(ADXL_X, INPUT);
  pinMode(ADXL_Y, INPUT);
  pinMode(ADXL_Z, INPUT);

  dht.begin();

  // Wait for sensors to stabilize
  delay(2000);
}

void loop() {
  // Read Temperature and Humidity
  float h = dht.readHumidity();
  float t = dht.readTemperature();

  // Read Light (raw LDR value, no lux conversion)
  int lightRaw = analogRead(LDR_PIN);

  // Read Accelerometer (raw analog values, do not assume 512 center)
  int ax = analogRead(ADXL_X);
  int ay = analogRead(ADXL_Y);
  int az = analogRead(ADXL_Z);

  // Read Rain (Digital 0 or 1, logical mapping handled in python)
  int rain = digitalRead(RAIN_PIN);

  // Check if any reads failed (specifically DHT)
  if (isnan(h) || isnan(t)) {
    // We send an error indicator or skip depending on protocol.
    // The spec requires ignoring bad readings gracefully, so we'll output
    // -999.0 for invalid DHT
    t = -999.0;
    h = -999.0;
  }

  // Output the strictly defined serial format:
  // TEMP=23.7,HUM=47.0,LIGHT=300,AX=385,AY=375,AZ=368,RAIN=1
  Serial.print("TEMP=");
  Serial.print(t, 1);
  Serial.print(",HUM=");
  Serial.print(h, 1);
  Serial.print(",LIGHT=");
  Serial.print(lightRaw);
  Serial.print(",AX=");
  Serial.print(ax);
  Serial.print(",AY=");
  Serial.print(ay);
  Serial.print(",AZ=");
  Serial.print(az);
  Serial.print(",RAIN=");
  Serial.println(rain);

  // Run loop at ~10Hz for responsive but stable readings
  delay(100);
}
