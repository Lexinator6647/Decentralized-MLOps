const fs = require("fs");

const getAccessToken = async () => {
  try {
    const response = await fetch("https://sandbox.api.o2-oracle.io/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email: "admin@team03.o2",
        password: "Phsz+HbM3un",
      }),
    });

    if (!response.ok) {
      throw new Error(
        `Failed to fetch access token. Status: ${response.status}`
      );
    }

    const data = await response.json();
    const { token, organizationId } = data;

    return { token, organizationId };
  } catch (error) {
    console.error("Error in getAccessToken:", error.message);
    throw error; // Rethrow so the caller can handle the error
  }
};

// Function to get AppID and AppName
const getAppIds = async (accessToken, organizationId) => {
  try {
    const response = await fetch(
      `https://sandbox.api.o2-oracle.io/apps/?organizationId=${organizationId}`,
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to fetch apps. Status: ${response.status}`);
    }

    const responseData = await response.json();

    // Check if responseData.data is an array and has elements
    if (Array.isArray(responseData.data) && responseData.data.length > 0) {
      const apps = responseData.data[0]; // Get the first app if available
      const appId = apps.id;
      const appName = apps.name;

      return { appId, appName };
    } else {
      throw new Error("No apps found or invalid response structure");
    }
  } catch (error) {
    console.error("Error in getAppIds:", error.message);
    throw error; // Rethrow so the caller can handle the error
  }
};

// Function to get propertyList IDs
const getPropertyListIds = async (appId, accessToken) => {
  try {
    const response = await fetch(
      `https://sandbox.api.o2-oracle.io/apps/${appId}/propertylists/`,
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
      }
    );

    if (!response.ok) {
      throw new Error(
        `Failed to fetch property lists. Status: ${response.status}`
      );
    }

    const propertyLists = await response.json();

    return propertyLists;
  } catch (error) {
    console.error("Error in getPropertyListIds:", error.message);
    throw error; // Rethrow so the caller can handle the error
  }
};

// Function to get propertyList rows
const getPropertyListRows = async (appId, propertylistId, accessToken) => {
  try {
    const response = await fetch(
      `https://sandbox.api.o2-oracle.io/apps/${appId}/propertylists/${propertylistId}/rows`,
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
      }
    );

    if (!response.ok) {
      throw new Error(
        `Failed to fetch property list rows. Status: ${response.status}`
      );
    }

    const propertylistRows = await response.json();

    return propertylistRows;
  } catch (error) {
    console.error("Error in getPropertyListRows:", error.message);
    throw error; // Rethrow so the caller can handle the error
  }
};

// Function to update propertyList rows
const postPropertyListRows = async (
  appId,
  propertylistId,
  accessToken,
  JSONContent
) => {
  try {
    const timestamp = new Date()
      .toISOString()
      .replace(/[-:.TZ]/g, "") // Remove unsafe characters
      .slice(0, 14);
    const payload = {
      rows: {
        [timestamp]: JSONContent,
      },
      operation: "create",
    };
    const response = await fetch(
      `https://sandbox.api.o2-oracle.io/apps/${appId}/propertylists/${propertylistId}/rows`,
      {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify(payload),
      }
    );

    if (!response.ok) {
      throw new Error(
        `Failed to post property list rows. Status: ${response.status}`
      );
    }

    const propertylistRows = await response.json();
    return propertylistRows;
  } catch (error) {
    console.error("Error in postPropertyListRows:", error.message);
    throw error; // Rethrow so the caller can handle the error
  }
};

// Function to publish propertyList changes
const publishChanges = async (appId, propertylistId, accessToken) => {
  try {
    const payload = {
      publishType: "all",
    };
    const response = await fetch(
      `https://sandbox.api.o2-oracle.io/apps/${appId}/propertylists/${propertylistId}/publish`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify(payload),
      }
    );

    if (!response.ok) {
      throw new Error(
        `Failed to publish propertyList changes. Status: ${response.status}`
      );
    }

    const propertylistRows = await response.json();
    return propertylistRows;
  } catch (error) {
    console.error("Error in publishChanges:", error.message);
    throw error; // Rethrow so the caller can handle the error
  }
};

const readJSON = async (filepath) => {
  return new Promise((resolve, reject) => {
    // Read the JSON file asynchronously
    fs.readFile(filepath, "utf8", (err, data) => {
      if (err) {
        reject("Error reading the file: " + err);
        return;
      }

      // Parse the JSON data
      const jsonData = JSON.parse(data);
      resolve(jsonData); // Resolve the promise with jsonData
    });
  });
};

// Main function to run the entire process
const main = async () => {
  try {
    // Get access token and organization ID
    const { token, organizationId } = await getAccessToken();

    // Get app details (appId and appName)
    const { appId, appName } = await getAppIds(token, organizationId);

    // Get property list IDs using the appId
    const propertylistIds = await getPropertyListIds(appId, token);

    const monitoring_metrics_prop_Id = propertylistIds.data[0].id;
    const training_metrics_prop_Id = propertylistIds.data[1].id;

    // Get property list IDs using the appId
    const training_metrics_listRows = await getPropertyListRows(
      appId,
      training_metrics_prop_Id,
      token
    );

    //Get training_metrics
    const train_metrics_filepath = "train_metrics.json";
    const train_metrics = await readJSON(train_metrics_filepath);

    //Get training_metrics
    const monitor_metrics_filepath = "monitor_metrics.json";
    const monitor_metrics = await readJSON(monitor_metrics_filepath);

    //Post PropertyList values
    const postPropertyList = await postPropertyListRows(
      appId,
      training_metrics_prop_Id,
      token,
      train_metrics
    );

    //Post PropertyList values
    const postMonitorList = await postPropertyListRows(
      appId,
      monitoring_metrics_prop_Id,
      token,
      monitor_metrics
    );

    //Publish training property list changes
    const publishTraningMetrics = await publishChanges(
      appId,
      training_metrics_prop_Id,
      token
    );

    //Publish monitering property list changes
    const publishMoniterMetrics = await publishChanges(
      appId,
      monitoring_metrics_prop_Id,
      token
    );
  } catch (error) {
    console.error("An error occurred in the main function:", error.message);
  }
};

main();
