#include "ProgressBar.h"

using namespace System;

ProgressBar::ProgressBar(int total, int barWidth, String^ prefix) {
	if (total <= 0)
		throw gcnew ArgumentOutOfRangeException("total", "Total must be a positive number.");

	this->total = total;
	this->barWidth = barWidth;
	this->prefix = prefix;
	this->startTime = DateTime::Now;

	Update(0);
}

void ProgressBar::Update(int progress) {
	if (progress < 0)
		throw gcnew ArgumentOutOfRangeException("progress", "Progress cannot be negative.");

	if (progress > total)
		progress = total;

	// i’»‚É•Ï‰»‚ª‚È‚¢ê‡‚ÍXV‚µ‚È‚¢
	if (this->current == progress)
		return;

	this->current = progress;

	double percentage = static_cast<double>(current) / total;
	int filledWidth = static_cast<int>(barWidth * percentage);

	TimeSpan elapsed = DateTime::Now - startTime;

	Console::Write("\r{0} [", prefix);
	for (int i = 0; i < filledWidth; i++)
		Console::Write("=");
	for (int i = filledWidth; i < barWidth; i++)
		Console::Write(" ");
	Console::Write("] {0,3}% ({1}/{2}) {3:hh\\:mm\\:ss}", static_cast<int>(percentage * 100), current, total, elapsed);

	Console::Out->Flush();
}

