var form = document.getElementById("select_pred_form");



// document.getElementById("show_pred").addEventListener("click", function () {
  
//   var formData = new FormData(form);
//   console.log("formData")
//   //print(form)
//   form.submit();
// });

function test(){
    

    var select_pred_days = document.getElementById("select_pred_days");
    window.location = window.location.origin + "/predict?n_days=" + select_pred_days.value;
    console.log(window.location.origin + "/predict?n_days="+select_pred_days.value);
    //print(form);
}