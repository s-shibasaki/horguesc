#include "MainForm.h"

using namespace System;
using namespace System::Windows::Forms;
using namespace dataloader;

[STAThread]
int main(array<String^>^ args)
{
    Application::EnableVisualStyles();
    Application::SetCompatibleTextRenderingDefault(false);

    // ���C���t�H�[���̍쐬�Ǝ��s
    Application::Run(gcnew MainForm());
    return 0;
}