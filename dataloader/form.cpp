#include "form.h"

[System::STAThreadAttribute]
int main(array<System::String^>^ args) {
	using namespace System;
	using namespace System::Windows::Forms;

	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false);
	Application::Run(gcnew dataloader::Form());

	return 0;
}
