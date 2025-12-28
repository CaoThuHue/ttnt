```
using System.Windows.Forms;
using System.Net;
using System.Runtime.Serialization.Json;
using System.IO;

        public void LoadDataGridView()
        {
            string link = "https://localhost:44326/api/sanpham";
            HttpWebRequest request = WebRequest.CreateHttp(link);
            WebResponse response = request.GetResponse();
  DataContractJsonSerializer js = new DataContractJsonSerializer (typeof (SanPham[]));
            object data = js.ReadObject(response.GetResponseStream());
            SanPham[] arr = data as SanPham[];
            dataGridView1.DataSource = arr;
        }
        public void LoadComboBox()
        {
            string link = "https://localhost:44326/api/danhmuc";
            HttpWebRequest request = WebRequest.CreateHttp(link);
            WebResponse response = request.GetResponse();
    DataContractJsonSerializer js = new DataContractJsonSerializer(typeof(DanhMuc[]));
            object data = js.ReadObject(response.GetResponseStream());
            DanhMuc[] arr1 = data as DanhMuc[];
            cboDanhMuc.DataSource = arr1;
            cboDanhMuc.ValueMember = "MaDanhMuc";
            cboDanhMuc.DisplayMember = "TenDanhMuc";
        }
        private void Form1_Load(object sender, EventArgs e)//Hien thi 
        {
            LoadDataGridView();
            LoadComboBox();
        }
        private void btThem_Click(object sender, EventArgs e)
        {
            string postString = string.Format("?ma={0}&ten={1}&gia={2}&madm={3}", txtMaSP.Text, txtTenSP.Text, txtDonGia.Text, cboDanhMuc.SelectedValue);
            string link = "https://localhost:44326/api/sanpham" + postString;
            HttpWebRequest request = WebRequest.CreateHttp(link);
            request.Method = "POST";
            Stream dataStream = request.GetRequestStream();
            DataContractJsonSerializer js = new DataContractJsonSerializer(typeof(bool));
            object data = js.ReadObject(request.GetResponse().GetResponseStream());
            bool kq = (bool)data;
            if (kq)
            {
                LoadDataGridView();
                MessageBox.Show(" them san pham thanh cong ");
            }
            else
            {
                MessageBox.Show(" them san pham that bai ");
            }
        }
        private void btSua_Click(object sender, EventArgs e)
        {
            string putString = string.Format("?ma={0}&ten={1}&gia={2}&madm={3}", txtMaSP.Text, txtTenSP.Text, txtDonGia.Text, cboDanhMuc.SelectedValue);
            string link = "https://localhost:44326/api/sanpham" + putString;
            HttpWebRequest request = WebRequest.CreateHttp(link);
            request.Method = "PUT";
            Stream dataStream = request.GetRequestStream();
            DataContractJsonSerializer js = new DataContractJsonSerializer(typeof(bool));
            object data = js.ReadObject(request.GetResponse().GetResponseStream());
            bool kq = (bool)data;
            if (kq)
            {
                LoadDataGridView();
                MessageBox.Show(" sua san pham thanh cong ");
            }
            else
            {
                MessageBox.Show(" sua san pham that bai ");
            }
        }

        private void btXoa_Click(object sender, EventArgs e)
        {
            string masp = txtMaSP.Text;
            string deleteS = string.Format("?id={0}", masp);
            string link = "https://localhost:44326/api/sanpham" + deleteS;
            HttpWebRequest request = WebRequest.CreateHttp(link);
            request.Method = "DELETE";
            DataContractJsonSerializer js = new DataContractJsonSerializer(typeof(bool));
            object data = js.ReadObject(request.GetResponse().GetResponseStream());
            bool kq = (bool)data;
            if (kq)
            {
                LoadDataGridView();
                MessageBox.Show(" xoa san pham thanh cong ");
            }
            else
            {
                MessageBox.Show(" xoa san pham that bai ");
            }   
        }
        private void btTim_Click(object sender, EventArgs e)
        {
            string madm = txtMaDM.Text;
            string link = "https://localhost:44326/api/sanpham?madm=" + madm;
            HttpWebRequest request = WebRequest.CreateHttp(link);
            WebResponse response = request.GetResponse();
            DataContractJsonSerializer js = new DataContractJsonSerializer(typeof(SanPham[]));
            object data = js.ReadObject(response.GetResponseStream());
            SanPham[] arr = data as SanPham[];
            dataGridView1.DataSource = arr;
        }
	private void dataGridView1_CellContentClick(object sender, DataGridViewCellEventArgs e)
        {
            int d = e.RowIndex;
            txtMaSP.Text = dataGridView1.Rows[d].Cells[0].Value.ToString();
            txtTenSP.Text = dataGridView1.Rows[d].Cells[1].Value.ToString();
            txtDonGia.Text = dataGridView1.Rows[d].Cells[2].Value.ToString();
            cboDanhMuc.Text = dataGridView1.Rows[d].Cells[3].Value.ToString();
        }
    }
}

using CaoThuHue_2023604963.Models;
        CSDLTestEntities db = new CSDLTestEntities();
        [HttpGet]//Lấy dữ liệu
        public List<SanPham> LaySP()  //đưa ra grid
        {
            return db.SanPhams.ToList();
        }
        [HttpGet]
        public List<SanPham> TimSPTheoDanhMuc(int madm)
        {
            return db.SanPhams.Where(x => x.MaDanhMuc == madm).ToList();
        }
        [HttpGet]
        public SanPham TimSPTheoMa(int ma)
        {
            return db.SanPhams.FirstOrDefault(x => x.Ma == ma);
        }
	[HttpGet]
public IEnumerable<SanPham> TimSPTheoKhoang(int min, int max)
{
    return db.SanPhams.Where(x => x.DonGia >= min && x.DonGia <= max).ToList();
}
        [HttpPost]//Thêm dữ liệu
        public bool ThemMoi(int ma, string ten, int gia, int madm)
        {
            SanPham sp = db.SanPhams.FirstOrDefault(x => x.Ma == ma);
            if (sp == null)
            {
                SanPham sp1 = new SanPham();
                sp1.Ma = ma;
                sp1.Ten = ten;
                sp1.DonGia = gia;
                sp1.MaDanhMuc = madm;
                db.SanPhams.Add(sp1);
                db.SaveChanges();
                return true;
            }
            return false;
        }
        [HttpPut]//Sửa dữ liệu
        public bool CapNhat(int ma, string ten, int gia, int madm)
        {
            SanPham sp = db.SanPhams.FirstOrDefault(x => x.Ma == ma);
            if (sp != null)
            {
                sp.Ma = ma;
                sp.Ten = ten;
                sp.DonGia = gia;
                sp.MaDanhMuc = madm;
                db.SaveChanges();
                return true;
            }
            return false;
        }
[HttpPut]  //Sửa từ nhiều cái A sang cái B, tạo 2 textbox A và B sau đó bấm sửa
public bool SuaGiaHangLoat(int giaCu, int giaMoi)
{
    try
    {
        // 1. Tìm danh sách tất cả sản phẩm đang có giá cũ
        var dsSanPham = db.SanPhams.Where(x => x.DonGia == giaCu).ToList();
        // Nếu không tìm thấy ai thì báo false
        if (dsSanPham.Count == 0) return false;
        // 2. Chạy vòng lặp để sửa giá từng cái một
        foreach (var sp in dsSanPham)
        {
            sp.DonGia = giaMoi;
        }
        // 3. Lưu tất cả thay đổi vào CSDL
        db.SaveChanges();
        return true;
    }
    catch
    {
        return false;
    }
}
        [HttpDelete]//Xóa dữ liệu
        public bool xoa(int id)
        {
            SanPham sp = db.SanPhams.FirstOrDefault(x => x.Ma == id);
            if (sp != null)
            {
                db.SanPhams.Remove(sp);
                db.SaveChanges();
                return true;
            }
            return false;
        }
[HttpDelete]//Xóa nhiều cái có cùng 1 kiểu dữ liệu
public bool xoa(int gia)
{
    try
    {
        var listSanPham = db.SanPhams.Where(x => x.DonGia == gia).ToList();
        if (listSanPham.Count == 0) return false;
        db.SanPhams.RemoveRange(listSanPham);
        db.SaveChanges();
        return true;
    }
    catch
    {
        return false;
    }
}
```



