#pragma once

namespace dataloader {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// form �̊T�v
	/// </summary>
	public ref class Form : public System::Windows::Forms::Form
	{
	public:
		Form(void)
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
		~Form()
		{
			if (components)
			{
				delete components;
			}
		}
	private: AxJVDTLabLib::AxJVLink^ axJVLink1;
	protected:

	private:
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
			System::ComponentModel::ComponentResourceManager^ resources = (gcnew System::ComponentModel::ComponentResourceManager(Form::typeid));
			this->axJVLink1 = (gcnew AxJVDTLabLib::AxJVLink());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->axJVLink1))->BeginInit();
			this->SuspendLayout();
			// 
			// axJVLink1
			// 
			this->axJVLink1->Enabled = true;
			this->axJVLink1->Location = System::Drawing::Point(13, 13);
			this->axJVLink1->Name = L"axJVLink1";
			this->axJVLink1->OcxState = (cli::safe_cast<System::Windows::Forms::AxHost::State^>(resources->GetObject(L"axJVLink1.OcxState")));
			this->axJVLink1->Size = System::Drawing::Size(192, 192);
			this->axJVLink1->TabIndex = 0;
			// 
			// Form
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(284, 261);
			this->Controls->Add(this->axJVLink1);
			this->Name = L"Form";
			this->Text = L"Dataloader";
			this->Load += gcnew System::EventHandler(this, &Form::Form_Load);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->axJVLink1))->EndInit();
			this->ResumeLayout(false);

		}
#pragma endregion
	private:
		System::Void Form_Load(System::Object^ sender, System::EventArgs^ e) {
		}
	};
}
