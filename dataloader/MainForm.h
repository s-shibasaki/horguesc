#pragma once

#include "DataLoader.h"

namespace dataloader {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// MainForm �̊T�v
	/// </summary>
	public ref class MainForm : public System::Windows::Forms::Form
	{
	public:
		MainForm(void)
		{
			InitializeComponent();
			//
			//TODO: �����ɃR���X�g���N�^�[ �R�[�h��ǉ����܂�
			//
		}

	protected:
		/// <summary>
		/// �g�p���̃��\�[�X�����ׂăN���[���A�b�v���܂��B
		/// </summary>
		~MainForm()
		{
			if (components)
			{
				delete components;
			}
		}

	private: 
		AxJVDTLabLib::AxJVLink^ jvlink;

		/// <summary>
		/// �K�v�ȃf�U�C�i�[�ϐ��ł��B
		/// </summary>
		System::ComponentModel::Container^ components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// �f�U�C�i�[ �T�|�[�g�ɕK�v�ȃ��\�b�h�ł��B���̃��\�b�h�̓��e��
		/// �R�[�h �G�f�B�^�[�ŕύX���Ȃ��ł��������B
		/// </summary>
		void InitializeComponent(void)
		{
			System::ComponentModel::ComponentResourceManager^ resources = (gcnew System::ComponentModel::ComponentResourceManager(MainForm::typeid));
			this->jvlink = (gcnew AxJVDTLabLib::AxJVLink());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->jvlink))->BeginInit();
			this->SuspendLayout();
			// 
			// jvlink
			// 
			this->jvlink->Enabled = true;
			this->jvlink->Location = System::Drawing::Point(13, 13);
			this->jvlink->Name = L"jvlink";
			this->jvlink->OcxState = (cli::safe_cast<System::Windows::Forms::AxHost::State^>(resources->GetObject(L"jvlink.OcxState")));
			this->jvlink->Size = System::Drawing::Size(192, 192);
			this->jvlink->TabIndex = 0;
			// 
			// MainForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(284, 261);
			this->Controls->Add(this->jvlink);
			this->Name = L"MainForm";
			this->Text = L"DataLoader";
			this->Load += gcnew System::EventHandler(this, &MainForm::MainForm_Load);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->jvlink))->EndInit();
			this->ResumeLayout(false);

		}
#pragma endregion

	private:
		System::Void MainForm_Load(System::Object^ sender, System::EventArgs^ e) {
			DataLoader^ dataLoader = gcnew DataLoader(jvlink);
			if (!dataLoader->Execute())
				Environment::Exit(1);
			Application::Exit();
		}
	};
}
