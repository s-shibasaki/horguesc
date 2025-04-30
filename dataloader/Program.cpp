#include "MainForm.h"

using namespace System;
using namespace System::Windows::Forms;
using namespace dataloader;

[STAThread]
int main(array<String^>^ args)
{
    Application::EnableVisualStyles();
    Application::SetCompatibleTextRenderingDefault(false);

    // メインフォームの作成と実行
    Application::Run(gcnew MainForm(args));
    return 0;
}