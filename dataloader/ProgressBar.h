#pragma once
ref class ProgressBar {
private:
	int total;
	int current;
	int barWidth;
	System::String^ prefix;
	System::DateTime startTime;

public:
	ProgressBar(int total, int barWidth, System::String^ prefix);
	void Update(int progress);
	int Increment();
};

