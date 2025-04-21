#include "form.h"

using namespace System;
using namespace System::Windows::Forms;

[System::STAThreadAttribute]
int main(array<String^>^ args) {
	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false);
	Application::Run(gcnew dataloader::Form());
	return 0;
}
