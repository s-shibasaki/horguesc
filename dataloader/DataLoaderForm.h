#pragma once

namespace dataloader {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	public ref class DataLoaderForm : public System::Windows::Forms::Form {
	public:
		DataLoaderForm(void) {
			InitializeComponent();
		}

		DataLoaderForm(array<String^>^ args) {
			InitializeComponent();
			args = args;
		}

	protected:
		~DataLoaderForm() {
			if (components)
			{
				delete components;
			}
		}

	private:
		array<String^>^ args;
		AxJVDTLabLib::AxJVLink^ jvlink;
		System::ComponentModel::Container^ components;
		System::Windows::Forms::Button^ button;
		System::Windows::Forms::TextBox^ textBox;

		void InitializeComponent(void) {

			System::ComponentModel::ComponentResourceManager^ resources = (gcnew System::ComponentModel::ComponentResourceManager(DataLoaderForm::typeid));
			this->jvlink = (gcnew AxJVDTLabLib::AxJVLink());
			this->button = (gcnew System::Windows::Forms::Button());
			this->textBox = (gcnew System::Windows::Forms::TextBox());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->jvlink))->BeginInit();
			this->SuspendLayout();
			
			this->jvlink->Enabled = true;
			this->jvlink->Location = System::Drawing::Point(13, 13);
			this->jvlink->Name = L"jvlink";
			this->jvlink->OcxState = (cli::safe_cast<System::Windows::Forms::AxHost::State^>(resources->GetObject(L"jvlink.OcxState")));
			this->jvlink->Size = System::Drawing::Size(192, 192);
			this->jvlink->TabIndex = 0;
			
			this->button->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((System::Windows::Forms::AnchorStyles::Bottom | System::Windows::Forms::AnchorStyles::Right));
			this->button->Location = System::Drawing::Point(696, 526);
			this->button->Name = L"button";
			this->button->Size = System::Drawing::Size(75, 23);
			this->button->TabIndex = 1;
			this->button->Text = L"button";
			this->button->UseVisualStyleBackColor = true;

			this->textBox->Anchor = static_cast<System::Windows::Forms::AnchorStyles>((((System::Windows::Forms::AnchorStyles::Top | System::Windows::Forms::AnchorStyles::Bottom)
				| System::Windows::Forms::AnchorStyles::Left)
				| System::Windows::Forms::AnchorStyles::Right));
			this->textBox->Location = System::Drawing::Point(13, 13);
			this->textBox->Multiline = true;
			this->textBox->Name = L"textBox";
			this->textBox->ScrollBars = System::Windows::Forms::ScrollBars::Both;
			this->textBox->Size = System::Drawing::Size(758, 507);
			this->textBox->TabIndex = 2;

			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(784, 561);
			this->Controls->Add(this->jvlink);
			this->Controls->Add(this->textBox);
			this->Controls->Add(this->button);
			this->Name = L"DataLoaderForm";
			this->Text = L"Data Loader";
			this->Load += gcnew System::EventHandler(this, &DataLoaderForm::DataLoaderForm_Load);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->jvlink))->EndInit();
			this->ResumeLayout(false);
		}

		System::Void DataLoaderForm_Load(System::Object^ sender, System::EventArgs^ e) {
			// TODO: コマンドライン引数を処理する

			int retVal;

			retVal = jvlink->JVInit("");
			if (retVal != 0) {
				textBox->AppendText("JVInit failed: " + retVal.ToString() + "\r\n");
				return;
			}
		}
	};
}
