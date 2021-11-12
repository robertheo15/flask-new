$("form[name=addAdmin]").submit(function (e) {
  var $form = $(this);
  var data = $form.serialize();

  $.ajax({
    url: "/admin/create-admin/",
    type: "POST",
    data: data,
    dataType: "json",
    success: function (resp) {
      window.location.href = "/admin/user/";
    },
    error: function (resp) {
      console.log(resp);
    },
  });
  e.preventDefault();
});

$("form[name=login_form]").submit(function (e) {
  var $form = $(this);
  var $error = $form.find(".error");
  var data = $form.serialize();

  $.ajax({
    url: "/user/login",
    type: "POST",
    data: data,
    dataType: "json",
    success: function (resp) {
      window.location.href = "/dashboard/";
    },
    error: function (resp) {
      console.log(resp);
      $error.text(resp.responseJSON.error).removeClass("error--hidden");
    },
  });

  e.preventDefault();
});
