export function get_myFunction(request) {
  let options = {
    "headers": {
      "Content-Type": "application/json"
    }
  };
  return wixData.query("Diagnostic Images")
    .eq("user image", request.path[0])
    .find()
    .then( (results) => {
      if(results.items.length > 0) {
        options.body = {
          "items": results.items
        };
        return ok(options);
      }
      options.body = {
        "error": `'${request.path[0]}' was not found`
      };
      return notFound(options);
    } )
    .catch( (error) => {
      options.body = {
        "error": error
      };
      return serverError(options);
    } );
}